"""
Qwen3-VL问答节点
用于基于图像的问答对话
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
    """Qwen3-VL视觉问答节点 - 支持基于图像的问答"""
    
    # 基础模型映射
    BASE_MODEL_REPO_MAP = {
        "Qwen3-VL-2B-Instruct": "Qwen/Qwen3-VL-2B-Instruct",
        "Qwen3-VL-2B-Thinking": "Qwen/Qwen3-VL-2B-Thinking",
        "Qwen3-VL-2B-Instruct-FP8": "Qwen/Qwen3-VL-2B-Instruct-FP8",
        "Qwen3-VL-2B-Thinking-FP8": "Qwen/Qwen3-VL-2B-Thinking-FP8",
        "Qwen3-VL-4B-Instruct": "Qwen/Qwen3-VL-4B-Instruct",
        "Qwen3-VL-4B-Thinking": "Qwen/Qwen3-VL-4B-Thinking",
        "Qwen3-VL-4B-Instruct-FP8": "Qwen/Qwen3-VL-4B-Instruct-FP8",
        "Qwen3-VL-4B-Thinking-FP8": "Qwen/Qwen3-VL-4B-Thinking-FP8",
        "Qwen3-VL-8B-Instruct": "Qwen/Qwen3-VL-8B-Instruct",
        "Qwen3-VL-8B-Thinking": "Qwen/Qwen3-VL-8B-Thinking",
        "Qwen3-VL-8B-Instruct-FP8": "Qwen/Qwen3-VL-8B-Instruct-FP8",
        "Qwen3-VL-8B-Thinking-FP8": "Qwen/Qwen3-VL-8B-Thinking-FP8"
    }
    
    @classmethod
    def get_available_models(cls):
        """获取可用的模型列表，包括基础模型和本地模型文件夹中的模型"""
        # 获取基础模型列表，并检查本地是否存在
        available_models = []
        models_dir = os.path.join(folder_paths.models_dir, "LLM", "Qwen-VL")
        
        # 检查基础模型是否已下载
        for model_name in cls.BASE_MODEL_REPO_MAP.keys():
            model_path = os.path.join(models_dir, model_name)
            if os.path.exists(model_path):
                available_models.append(f"{model_name}（已下载）")
            else:
                available_models.append(model_name)
        
        # 扫描本地模型文件夹，添加未在基础模型列表中的模型
        if os.path.exists(models_dir):
            for item in os.listdir(models_dir):
                item_path = os.path.join(models_dir, item)
                # 清理模型名称（移除可能的"（已下载）"标记）
                clean_item = item.replace("（已下载）", "")
                # 检查是否为目录且不在基础模型列表中
                if os.path.isdir(item_path) and clean_item not in cls.BASE_MODEL_REPO_MAP.keys():
                    # 检查是否已添加过（避免重复）
                    if not any(clean_item == m.replace("（已下载）", "") for m in available_models):
                        available_models.append(f"{item}（已下载）")
        
        return available_models
    
    # 动态模型映射，结合基础模型和本地模型
    @classmethod
    def get_model_repo_map(cls):
        """获取模型仓库映射，包括基础模型和本地模型"""
        model_repo_map = cls.BASE_MODEL_REPO_MAP.copy()
        
        # 扫描本地模型文件夹
        models_dir = os.path.join(folder_paths.models_dir, "LLM", "Qwen-VL")
        if os.path.exists(models_dir):
            for item in os.listdir(models_dir):
                item_path = os.path.join(models_dir, item)
                # 提取干净的模型名称（去除"（已下载）"标记）
                clean_item = item.replace("（已下载）", "")
                # 检查是否为目录且不在映射中
                if os.path.isdir(item_path) and clean_item not in model_repo_map:
                    # 对于本地模型，使用目录名作为仓库ID
                    model_repo_map[clean_item] = clean_item
        
        return model_repo_map
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "question": ("STRING", {
                    "default": "请描述这张图片的内容。",
                    "multiline": True,
                    "description": "输入问题"
                }),
                "chat_history": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "description": "对话历史记录（JSON格式）"
                }),
                "max_tokens": ("INT", {
                    "default": 512,
                    "min": 50,
                    "max": 2048,
                    "step": 50,
                    "description": "最大生成令牌数"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "description": "生成温度"
                })
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "default": "你是一个专业的视觉问答助手，能够准确回答关于图像内容的问题。",
                    "multiline": True,
                    "description": "系统提示词，用于设定AI角色和行为"
                }),
                "model_name": (cls.get_available_models(), {
                    "default": "Qwen3-VL-2B-Instruct",
                    "description": "选择Qwen3-VL模型版本"
                }),
                "device": (["Auto", "cuda", "cpu", "mps"], {
                    "default": "Auto",
                    "description": "设备选择"
                }),
                "quantization": (["无（FP16）", "4-bit", "8-bit"], {
                    "default": "无（FP16）",
                    "description": "模型量化选项"
                }),
                "attention_type": (["Eager: 最佳兼容性", "SDPA: 平衡", "Flash Attention 2: 最佳性能"], {
                    "default": "SDPA: 平衡",
                    "description": "注意力机制类型"
                }),
                "clear_model_cache": ("BOOLEAN", {
                    "default": False,
                    "description": "是否清除模型缓存"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "description": "随机种子，-1表示随机"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("答案", "更新历史")
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
        if device_name == "Auto":
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
            print(f"[Qwen3-VL QA] 正在下载模型 {model_name}...")
            model_path = os.path.join(folder_paths.models_dir, "LLM", "Qwen-VL", model_name)
            
            snapshot_download(
                repo_id=repo_id,
                local_dir=model_path,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            print(f"[Qwen3-VL QA] 模型下载完成: {model_path}")
            return model_path
            
        except Exception as e:
            print(f"[Qwen3-VL QA] 模型下载失败: {str(e)}")
            print("[Qwen3-VL QA] 解决方案:")
            print("1. 检查网络连接")
            print("2. 确认HuggingFace访问权限")
            print("3. 手动下载模型到指定路径")
            raise e
    
    def load_model(self, model_name, device, attention_type, quantization):
        """加载模型，支持独立下载和缓存机制"""
        # 提取干净的模型名称（去除"（已下载）"标记）
        clean_model_name = model_name.replace("（已下载）", "")
        
        try:
            # 获取设备
            device_name = self.get_device(device)
            
            # 检查是否需要重新加载模型
            if (self.model is not None and 
                self.current_model_name == model_name and 
                self.current_device == device_name and 
                self.current_attention_type == attention_type):
                print(f"[Qwen3-VL QA] 使用缓存模型: {model_name}")
                return self.model, self.processor
            
            # 清理之前的模型缓存
            if self.model is not None:
                print(f"[Qwen3-VL QA] 清理之前的模型缓存...")
                del self.model
                del self.processor
                torch.cuda.empty_cache()
            
            # 获取模型映射
            model_repo_map = self.get_model_repo_map()
            
            # 从映射中获取仓库ID，使用干净的模型名称
            repo_id = model_repo_map.get(clean_model_name)
            if not repo_id:
                raise ValueError(f"[Qwen3-VL QA] 不支持的模型名称: {clean_model_name}")
            
            # 使用干净的模型名称作为本地目录名
            model_checkpoint = os.path.join(folder_paths.models_dir, "LLM", "Qwen-VL", clean_model_name)
            
            # 对于HuggingFace模型（包含"/"），需要下载
            if "/" in repo_id:
                # 检查模型是否已存在
                if not os.path.exists(model_checkpoint):
                    print(f"[Qwen3-VL QA] 本地模型不存在，开始下载: {model_name}")
                    model_checkpoint = self._download_model_with_progress(model_name, repo_id)
                else:
                    print(f"[Qwen3-VL QA] 使用本地模型: {model_checkpoint}")
            else:
                # 对于本地模型，检查是否存在
                if not os.path.exists(model_checkpoint):
                    raise ValueError(f"[Qwen3-VL QA] 本地模型不存在: {model_checkpoint}")
            
            # 注意力类型映射
            attention_map = {
                "Eager: 最佳兼容性": "eager",
                "SDPA: 平衡": "sdpa", 
                "Flash Attention 2: 最佳性能": "flash_attention_2"
            }
            attention_impl = attention_map.get(attention_type, "eager")
            
            # 量化配置
            quantization_config = None
            if quantization == "4-bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif quantization == "8-bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            print(f"[Qwen3-VL QA] 加载模型: {model_name}")
            print(f"[Qwen3-VL QA] 设备: {device_name}")
            print(f"[Qwen3-VL QA] 量化: {quantization}")
            print(f"[Qwen3-VL QA] 注意力类型: {attention_type}")
            
            # 模型加载配置
            model_kwargs = {
                "dtype": torch.float16,
                "device_map": "auto" if device_name == "auto" or quantization != "None" else None,
                "attn_implementation": attention_impl,
            }
            
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config
            
            # 加载模型和处理器
            self.processor = AutoProcessor.from_pretrained(model_checkpoint)
            self.model = AutoModelForVision2Seq.from_pretrained(model_checkpoint, **model_kwargs)
            
            if device_name != "auto" and quantization == "None":
                self.model = self.model.to(device_name)
            
            # 更新当前状态
            self.current_model_name = model_name
            self.current_device = device_name
            self.current_attention_type = attention_type
            self.current_quantization = quantization
            
            print(f"[Qwen3-VL QA] 模型加载成功: {model_name}")
            return self.model, self.processor
            
        except Exception as e:
            print(f"[Qwen3-VL QA] 模型加载失败: {str(e)}")
            print("[Qwen3-VL QA] 故障排除建议:")
            print("1. 检查模型文件是否完整")
            print("2. 确认模型路径配置是否正确")
            print("3. 检查设备内存是否充足")
            print("4. 尝试使用量化模式或切换注意力类型")
            raise e
    
    def generate_answer(self, image, question, chat_history, max_tokens, temperature, 
                       system_prompt="", seed=-1, clear_model_cache=False, model_name="Qwen3-VL-2B-Instruct", 
                       device="Auto", quantization="无（FP16）", 
                       attention_type="SDPA: 平衡"):
        """根据图像和问题生成答案，支持对话历史"""
        
        # 清理模型缓存（如果需要）
        if clear_model_cache:
            print("[Qwen3-VL QA] 清理模型缓存")
            self.cleanup_model()
        
        # 加载模型
        current_model, current_processor = self.load_model(model_name, device, attention_type, quantization)
        
        # 处理输入图像
        if isinstance(image, torch.Tensor):
            # 将张量转换为PIL图像
            if image.dim() == 4:  # (B, H, W, C)
                image = image[0]  # 取第一张图像
            if image.dim() == 3:  # (H, W, C)
                image = image.cpu().numpy()
            
            # 确保正确的数值范围
            if image.dtype == np.float32:
                image = (image * 255).astype(np.uint8)
            
            image = Image.fromarray(image)
        
        # 将PIL图像转换为base64编码以避免JSON序列化问题
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_buffer.seek(0)
        img_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
        image_base64 = f"data:image/png;base64,{img_b64}"
        
        # 构建对话历史
        messages = []
        
        # 添加系统提示词（如果提供）
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # 解析对话历史
        if chat_history:
            try:
                history_data = json.loads(chat_history)
                if isinstance(history_data, list):
                    messages.extend(history_data)
            except:
                # 如果解析失败，忽略对话历史
                pass
        
        # 添加当前问题
        messages.append({
            "role": "user", 
            "content": [
                {"type": "image", "image": image_base64},
                {"type": "text", "text": question}
            ]
        })
        
        # 获取设备信息
        device = next(current_model.parameters()).device
        
        # 准备输入 - 使用与集成模块相同的方法
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
            # 回退到传统方法
            text = current_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = current_processor(
                text=[text],
                images=[image],
                padding=True,
                return_tensors="pt"
            )
        
        if device != "auto":
            inputs = inputs.to(device)
        
        # 设置随机种子
        if seed != -1:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
        
        # 生成答案
        with torch.no_grad():
            generated_ids = current_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.95,
                pad_token_id=current_processor.tokenizer.eos_token_id
            )
        
        # 解码生成的文本
        generated_text = current_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # 提取答案部分
        answer = generated_text.split("assistant")[-1].strip() if "assistant" in generated_text else generated_text
        
        # 更新对话历史
        messages.append({"role": "assistant", "content": answer})
        
        # 序列化更新后的历史
        updated_history = json.dumps(messages, ensure_ascii=False, indent=2)
        
        print(f"[Qwen3-VL QA] 答案: {answer[:100]}...")
        
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
        
        # Clean up GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("[Qwen3-VL-QA] Model cache cleared")


# 节点映射
NODE_CLASS_MAPPINGS = {
    "Qwen3VLQA": Qwen3VLQA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3VLQA": "Qwen3-VL视觉问答",
}