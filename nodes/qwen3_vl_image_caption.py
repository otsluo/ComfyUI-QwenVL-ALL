import torch
import os
import json
import folder_paths
from PIL import Image
import numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers import BitsAndBytesConfig
from huggingface_hub import snapshot_download
import gc


class Qwen3VLImageCaption:
    """Qwen3-VL图片反推节点 - 专门用于图像描述生成"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.current_model_name = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "反推提示词": ("STRING", {
                    "default": "请详细描述这张图片的内容，包括主要物体、场景、颜色、风格等特征。",
                    "multiline": True
                }),
                "系统提示词": ("STRING", {
                    "default": "你是一个专业的图像分析助手，能够准确描述图片中的各种细节。",
                    "multiline": True
                }),
                "最大令牌数": ("INT", {
                    "default": 512,
                    "min": 50,
                    "max": 2048,
                    "step": 50
                }),
                "温度": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1
                }),

            },
            "optional": {
                "模型名称": (list(cls.MODEL_REPO_MAP.keys()), {
                    "default": "Qwen3-VL-2B-Instruct"
                }),
                "设备": (["自动", "cuda", "cpu", "mps"], {
                    "default": "自动"
                }),
                "量化": (["无", "4位量化", "8位量化"], {
                    "default": "4位量化"
                }),
                "注意力类型": (["标准注意力：兼容性好", "SDPA注意力：平衡", "Flash注意力2：性能优先"], {
                    "default": "标准注意力：兼容性好"
                }),
                "清除缓存": ("BOOLEAN", {
                    "default": False
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("图像描述", "详细响应")
    FUNCTION = "generate_caption"
    CATEGORY = "QwenVL-ALL"
    
    MODEL_REPO_MAP = {
        "Qwen3-VL-2B-Instruct": "Qwen/Qwen3-VL-2B-Instruct",
        "Qwen3-VL-4B-Instruct": "Qwen/Qwen3-VL-4B-Instruct"
    }
    
    def get_device(self, device_preference):
        """获取设备"""
        if device_preference == "自动":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device_preference.lower()
    
    def _download_model_with_progress(self, model_id: str, local_dir: str):
        """下载模型并显示进度"""
        print(f"\n{'='*70}")
        print(f"[Qwen3-VL图像反推] 📥 开始下载模型: {model_id}")
        print(f"[Qwen3-VL图像反推] 📂 保存路径: {local_dir}")
        print(f"[Qwen3-VL图像反推] ⏳ 请耐心等待，下载可能需要几分钟到几十分钟...")
        print(f"[Qwen3-VL图像反推] 💡 下载进度将在下方显示")
        print(f"{'='*70}\n")

        try:
            # snapshot_download 会自动显示每个文件的下载进度条
            snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            print(f"\n{'='*70}")
            print(f"[Qwen3-VL图像反推] ✅ 模型下载完成: {model_id}")
            print(f"{'='*70}\n")
        except Exception as e:
            print(f"\n{'='*70}")
            print(f"[Qwen3-VL图像反推] ❌ 模型下载失败: {e}")
            print(f"[Qwen3-VL图像反推] 💡 解决方法:")
            print(f"[Qwen3-VL图像反推]    1. 检查网络连接")
            print(f"[Qwen3-VL图像反推]    2. 使用镜像站: export HF_ENDPOINT=https://hf-mirror.com")
            print(f"[Qwen3-VL图像反推]    3. 使用代理: export HTTP_PROXY=http://127.0.0.1:7890")
            print(f"[Qwen3-VL图像反推]    4. 手动下载模型到: {local_dir}")
            print(f"{'='*70}\n")
            raise
    
    def load_model(self, model_name, device, quantization="无", attention_type="标准注意力：兼容性好"):
        """加载模型"""
        if self.current_model_name == model_name and self.model is not None:
            return
            
        # 清理之前的模型
        self.cleanup_model()
        
        # 从映射中获取HuggingFace仓库ID
        repo_id = self.MODEL_REPO_MAP.get(model_name)
        if not repo_id:
            raise ValueError(f"不支持的模型: {model_name}")
        
        # 使用模型名称作为本地目录名称
        model_checkpoint = os.path.join(
            folder_paths.models_dir, "LLM", "Qwen-VL", model_name
        )
        
        # 如果模型不存在则下载
        if not os.path.exists(model_checkpoint):
            self._download_model_with_progress(repo_id, model_checkpoint)
        
        try:
            print(f"[Qwen3-VL图像反推] 加载模型: {model_name} 到设备: {device}")
            
            # 设置模型加载参数
            model_kwargs = {
                "dtype": torch.float16 if device == "cuda" else torch.float32,
                "device_map": "auto" if device == "cuda" else None,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True
            }
            
            # 配置量化
            quantization_config = None
            if quantization == "4位量化":
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
                model_kwargs["quantization_config"] = quantization_config
            elif quantization == "8位量化":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                model_kwargs["quantization_config"] = quantization_config
            
            # 配置注意力类型
            attention_type_map = {
                "标准注意力：兼容性好": "eager",
                "SDPA注意力：平衡": "sdpa", 
                "Flash注意力2：性能优先": "flash_attention_2"
            }
            attn_implementation = attention_type_map.get(attention_type, "eager")
            model_kwargs["attn_implementation"] = attn_implementation
            
            # 从本地目录加载处理器和模型
            self.processor = AutoProcessor.from_pretrained(
                model_checkpoint,
                trust_remote_code=True
            )
            
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_checkpoint,
                **model_kwargs
            )
            
            if device != "cuda":
                self.model = self.model.to(device)
                
            self.current_model_name = model_name
            print(f"[Qwen3-VL图像反推] 模型加载成功: {model_name}")
            
        except Exception as e:
            print(f"[Qwen3-VL图像反推] 模型加载失败: {str(e)}")
            raise RuntimeError(f"模型加载失败: {str(e)}。请确保网络连接正常，或手动下载模型到 ComfyUI/models/LLM/Qwen-VL/ 目录")
    
    def cleanup_model(self):
        """清理模型缓存"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self.current_model_name = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def tensor_to_pil(self, tensor):
        """将Tensor转换为PIL图像"""
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        if tensor.dim() == 3 and tensor.shape[2] == 3:
            tensor = tensor.permute(2, 0, 1)
        
        # 转换为numpy数组
        if tensor.max() <= 1.0:
            tensor = tensor * 255.0
        
        tensor = tensor.clamp(0, 255)
        array = tensor.cpu().numpy().astype(np.uint8)
        
        # 确保正确的形状
        if array.shape[0] == 3:
            array = array.transpose(1, 2, 0)
        
        return Image.fromarray(array)
    
    def generate_caption(self, 图像, 反推提示词, 系统提示词, 最大令牌数, 温度, 模型名称="Qwen3-VL-2B-Instruct", 设备="自动", 量化="4位量化", 注意力类型="标准注意力：兼容性好", 清除缓存=False):
        """生成图像描述"""
        
        print(f"[Qwen3-VL] 调试信息 - 传入参数:")
        print(f"  - 清除缓存: {清除缓存}")
        print(f"  - 当前缓存状态: 模型={self.model is not None}, 处理器={self.processor is not None}")
        
        if 清除缓存:
            self.cleanup_model()
        

        
        try:
            # 独立加载模式 - 使用原来的加载方式
            print(f"[Qwen3-VL] 使用独立加载模式: {模型名称}")
            try:
                device_actual = self.get_device(设备)
                self.load_model(模型名称, device_actual, 注意力类型, 量化)
                current_model = self.model
                current_processor = self.processor
                print(f"[Qwen3-VL图像反推] 模型加载成功")
            except Exception as load_error:
                print(f"[Qwen3-VL图像反推] 模型加载失败: {str(load_error)}")
                raise RuntimeError(f"模型加载失败: {str(load_error)}。请确保网络连接正常，或手动下载模型到 ComfyUI/models/LLM/Qwen-VL/ 目录")
            
            # 转换图像
            pil_image = self.tensor_to_pil(图像)
            
            # 准备消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": 反推提示词}
                    ]
                }
            ]
            
            # 应用聊天模板
            text = current_processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # 处理输入
            inputs = current_processor(
                text=[text],
                images=[pil_image],
                return_tensors="pt",
                padding=True
            )
            
            # 获取设备信息
            device = next(current_model.parameters()).device
            
            # 移动到相应设备
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # 生成描述
            print(f"[Qwen3-VL图像反推] 开始生成图像描述...")
            
            with torch.no_grad():
                generated_ids = current_model.generate(
                    **inputs,
                    max_new_tokens=最大令牌数,
                    temperature=温度,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=current_processor.tokenizer.pad_token_id,
                    eos_token_id=current_processor.tokenizer.eos_token_id
                )
            
            # 解码响应
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            
            response_text = current_processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]
            
            print(f"[Qwen3-VL图像反推] 图像描述生成完成")
            
            # 提取纯描述文本（移除思考过程标签）
            import re
            clean_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
            
            return (clean_response, response_text)
            
        except Exception as e:
            error_msg = f"图像描述生成失败: {str(e)}"
            print(f"[Qwen3-VL图像反推] {error_msg}")
            return (error_msg, f"错误详情: {str(e)}")


# 节点映射
NODE_CLASS_MAPPINGS = {
    "Qwen3VLImageCaption": Qwen3VLImageCaption,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3VLImageCaption": "Qwen3-VL图像反推",
}