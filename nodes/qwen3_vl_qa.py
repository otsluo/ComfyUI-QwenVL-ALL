"""
Qwen3-VL 问答节点
用于基于图像进行问答对话
"""

import torch
import numpy as np
from PIL import Image
import folder_paths
import os
import json
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from huggingface_hub import snapshot_download
import random
import io
import base64
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None


class Qwen3VLQA:
    """Qwen3-VL 视觉问答节点 - 支持基于图像进行问答对话"""
    
    # 模型ID映射：模型名称 -> HuggingFace仓库ID
    MODEL_REPO_MAP = {
        "Qwen3-VL-2B-Instruct": "Qwen/Qwen3-VL-2B-Instruct",
        "Qwen3-VL-4B-Instruct": "Qwen/Qwen3-VL-4B-Instruct"
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "图像": ("IMAGE",),
                "问题": ("STRING", {"default": "这张图片里有什么？", "multiline": True}),
                "系统提示词": ("STRING", {"default": "你是一个智能助手，能够准确回答关于图像的问题。请用中文回答。", "multiline": True}),
                "最大令牌数": ("INT", {"default": 512, "min": 50, "max": 2048, "step": 50}),
                "温度": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 2.0, "step": 0.1}),
            },
            "optional": {
                "历史对话": ("STRING", {"default": "", "multiline": True}),
                "清理模型缓存": ("BOOLEAN", {"default": False, "description": "清理模型缓存，重新加载模型"}),
                "模型名称": (list(cls.MODEL_REPO_MAP.keys()), {"default": "Qwen3-VL-2B-Instruct"}),
                "设备": (["自动", "cuda", "cpu", "mps"], {"default": "自动"}),
                "量化": (["无", "4位量化", "8位量化"], {"default": "4位量化"}),
                "注意力类型": (["标准注意力：兼容性好", "SDPA注意力：平衡", "Flash注意力2：性能优先"], {"default": "标准注意力：兼容性好"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("回答", "更新历史")
    FUNCTION = "generate_answer"
    CATEGORY = "QwenVL-ALL"
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.current_model_name = None
        self.current_device = None
        self.current_attention_type = None
        self.current_quantization = None
    
    def get_device(self, device_name):
        """获取设备类型"""
        if device_name == "自动":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device_name
    
    def _download_model_with_progress(self, model_name, repo_id):
        """下载模型到本地路径"""
        try:
            print(f"[Qwen3-VL问答] 正在下载模型 {model_name}...")
            model_path = os.path.join(folder_paths.models_dir, "LLM", "Qwen-VL", model_name)
            
            snapshot_download(
                repo_id=repo_id,
                local_dir=model_path,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            print(f"[Qwen3-VL问答] 模型下载完成: {model_path}")
            return model_path
            
        except Exception as e:
            print(f"[Qwen3-VL问答] 模型下载失败: {str(e)}")
            print("[Qwen3-VL问答] 解决方法建议:")
            print("1. 检查网络连接")
            print("2. 确认HuggingFace访问权限")
            print("3. 手动下载模型到指定路径")
            raise e
    
    def load_model(self, model_name, device, attention_type, quantization):
        """加载模型，支持独立下载和缓存机制"""
        try:
            # 获取设备
            device_name = self.get_device(device)
            
            # 检查是否需要重新加载模型
            if (self.model is not None and 
                self.current_model_name == model_name and 
                self.current_device == device_name and 
                self.current_attention_type == attention_type):
                print(f"[Qwen3-VL问答] 使用缓存模型: {model_name}")
                return self.model, self.processor
            
            # 清理之前的模型缓存
            if self.model is not None:
                print(f"[Qwen3-VL问答] 清理之前的模型缓存...")
                del self.model
                del self.processor
                torch.cuda.empty_cache()
            
            # 获取模型仓库ID
            if model_name not in self.MODEL_REPO_MAP:
                raise ValueError(f"[Qwen3-VL问答] 不支持的模型名称: {model_name}")
            
            repo_id = self.MODEL_REPO_MAP[model_name]
            
            # 构建本地模型路径
            model_checkpoint = os.path.join(folder_paths.models_dir, "LLM", "Qwen-VL", model_name)
            
            # 检查模型是否已存在
            if not os.path.exists(model_checkpoint):
                print(f"[Qwen3-VL问答] 本地模型不存在，开始下载: {model_name}")
                model_checkpoint = self._download_model_with_progress(model_name, repo_id)
            else:
                print(f"[Qwen3-VL问答] 使用本地模型: {model_checkpoint}")
            
            # 注意力类型映射
            attention_map = {
                "标准注意力：兼容性好": "eager",
                "SDPA注意力：平衡": "sdpa", 
                "Flash注意力2：性能优先": "flash_attention_2"
            }
            attention_impl = attention_map.get(attention_type, "eager")
            
            # 量化配置
            quantization_config = None
            if quantization == "4位量化":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif quantization == "8位量化":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            print(f"[Qwen3-VL问答] 正在加载模型: {model_name}")
            print(f"[Qwen3-VL问答] 设备: {device_name}")
            print(f"[Qwen3-VL问答] 量化: {quantization}")
            print(f"[Qwen3-VL问答] 注意力类型: {attention_type}")
            
            # 模型加载配置
            model_kwargs = {
                "dtype": torch.float16,
                "device_map": "auto" if device_name == "auto" or quantization != "无" else None,
                "attn_implementation": attention_impl,
            }
            
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config
            
            # 加载模型和处理器
            self.processor = AutoProcessor.from_pretrained(model_checkpoint)
            self.model = AutoModelForVision2Seq.from_pretrained(model_checkpoint, **model_kwargs)
            
            if device_name != "auto" and quantization == "无":
                self.model = self.model.to(device_name)
            
            # 更新当前状态
            self.current_model_name = model_name
            self.current_device = device_name
            self.current_attention_type = attention_type
            self.current_quantization = quantization
            
            print(f"[Qwen3-VL问答] 模型加载完成: {model_name}")
            return self.model, self.processor
            
        except Exception as e:
            print(f"[Qwen3-VL问答] 模型加载失败: {str(e)}")
            print("[Qwen3-VL问答] 解决方法建议:")
            print("1. 检查模型文件是否完整")
            print("2. 确认模型路径配置正确")
            print("3. 检查设备内存是否充足")
            print("4. 尝试使用量化模式或切换注意力类型")
            raise e
    
    def generate_answer(self, 图像, 问题, 系统提示词, 最大令牌数, 温度,
                       清理模型缓存=False, 历史对话="", 模型名称="Qwen3-VL-2B-Instruct", 
                       设备="自动", 量化="4位量化", 注意力类型="标准注意力：兼容性好"):
        """基于图像生成回答"""
        

        
        # 清理模型缓存（如果需要）
        if 清理模型缓存:
            print("[Qwen3-VL问答] 清理模型缓存")
            self.cleanup_model()
        
        # 加载模型
        current_model, current_processor = self.load_model(模型名称, 设备, 注意力类型, 量化)
        
        # 处理输入图像
        if isinstance(图像, torch.Tensor):
            # 将张量转换为PIL图像
            if 图像.dim() == 4:  # (B, H, W, C)
                图像 = 图像[0]  # 取第一张图像
            if 图像.dim() == 3:  # (H, W, C)
                图像 = 图像.cpu().numpy()
            
            # 确保数值范围正确
            if 图像.dtype == np.float32:
                图像 = (图像 * 255).astype(np.uint8)
            
            图像 = Image.fromarray(图像)
        
        # 将PIL图像转换为base64编码，避免JSON序列化问题
        img_buffer = io.BytesIO()
        图像.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        image_base64 = f"data:image/png;base64,{img_b64}"
        
        # 构建对话历史
        messages = []
        if 系统提示词:
            messages.append({"role": "system", "content": 系统提示词})
        
        # 解析历史对话
        if 历史对话:
            try:
                history_data = json.loads(历史对话)
                if isinstance(history_data, list):
                    messages.extend(history_data)
            except:
                # 如果解析失败，忽略历史对话
                pass
        
        # 添加当前问题
        messages.append({
            "role": "user", 
            "content": [
                {"type": "image", "image": image_base64},
                {"type": "text", "text": 问题}
            ]
        })
        
        # 获取设备信息
        device = next(current_model.parameters()).device
        
        # 准备输入 - 使用与整合模块相同的方式
        if process_vision_info is not None:
            # 使用qwen_vl_utils处理视觉信息
            text = current_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs = process_vision_info(
                messages,
                return_video_kwargs=True
            )
            
            inputs = current_processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **video_kwargs
            )
        else:
            # 回退到传统方式
            text = current_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = current_processor(
                text=[text],
                images=[图像],
                padding=True,
                return_tensors="pt"
            )
        
        if device != "auto":
            inputs = inputs.to(device)
        
        # 生成回答
        with torch.no_grad():
            generated_ids = current_model.generate(
                **inputs,
                max_new_tokens=最大令牌数,
                temperature=温度,
                do_sample=温度 > 0,
                top_p=0.95,
                pad_token_id=current_processor.tokenizer.eos_token_id
            )
        
        # 解码生成的文本
        generated_text = current_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 提取回答部分
        answer = generated_text.split("assistant")[-1].strip() if "assistant" in generated_text else generated_text
        
        # 更新对话历史
        messages.append({"role": "assistant", "content": answer})
        
        # 序列化更新后的历史
        updated_history = json.dumps(messages, ensure_ascii=False, indent=2)
        
        print(f"[Qwen3-VL问答] 回答: {answer[:100]}...")
        
        return (answer, updated_history)
    
    def cleanup_model(self):
        """清理模型缓存"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        self.current_model_name = None
        self.current_device = None
        self.current_attention_type = None
        self.current_quantization = None
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("[Qwen3-VL-QA] 模型缓存已清理")


# 节点映射
NODE_CLASS_MAPPINGS = {
    "Qwen3VLQA": Qwen3VLQA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3VLQA": "Qwen3-VL 视觉问答",
}