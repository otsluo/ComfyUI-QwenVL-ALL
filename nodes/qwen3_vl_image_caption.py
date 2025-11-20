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
    """Qwen3-VL图像描述节点 - 专门用于生成图像描述"""
    
    # 提示预设字典
    CAPTION_PRESETS = {
        "提示风格 - 标签": "你的任务是根据图像中的视觉信息，为文本到图像AI生成一个简洁的逗号分隔标签列表。将输出限制在最多50个独特标签。严格描述视觉元素，如主体、服装、环境、颜色、光线和构图。不要包含抽象概念、解释、营销术语或技术术语（例如，不要包含'SEO'、'品牌对齐'、'病毒式传播潜力'）。目标是简洁的视觉描述符列表。避免重复标签。",
        "提示风格 - 简洁": "分析图像并生成一个简单的单句文本到图像提示。简洁地描述主要主体和环境。",
        "提示风格 - 详细": "基于图像生成一个详细的、艺术性的文本到图像提示。将主体、动作、环境、光线和整体氛围结合成一个连贯的段落，大约2-3句话。专注于关键的视觉细节。",
        "提示风格 - 极详细": "从图像生成一个极其详细和描述性的文本到图像提示。创建一个丰富的段落，详细阐述主体的外观、服装纹理、特定的背景元素、光线的质量和颜色、阴影和整体氛围。目标是高度描述性和沉浸式的提示。",
        "提示风格 - 电影感": "作为大师级提示工程师。为图像生成AI创建一个高度详细和富有表现力的提示。描述主体、姿势、环境、光线、氛围和艺术风格（例如，照片级真实、电影感、绘画风格）。将所有元素编织成一个自然的语言段落，专注于视觉冲击。",
        "创意 - 详细分析": "详细描述这张图像，将主体、服装、配饰、背景和构图分解为独立的部分。",
        "创意 - 视频总结": "总结这个视频中的关键事件和叙事要点。",
        "创意 - 短篇故事": "根据这张图像或视频写一个富有想象力的短篇故事。",
        "创意 - 优化扩展提示": "优化和增强以下用户提示，用于创意文本到图像生成。保持含义和关键词，使其更具表现力和视觉丰富性。仅输出改进后的提示文本本身，不要有任何推理步骤、思考过程或额外评论。"
    }
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.current_model_name = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "caption_prompt": ("STRING", {
                    "default": "请详细描述这张图片，包括主要对象、场景、颜色、风格和其他特征。",
                    "multiline": True
                }),
                "preset_selection": (["无预设"] + list(cls.CAPTION_PRESETS.keys()), {
                    "default": "无预设"
                }),
                "system_prompt": ("STRING", {
                    "default": "您是一位专业的图像分析助手，能够准确描述图像中的各种细节。",
                    "multiline": True
                }),
                "max_tokens": ("INT", {
                    "default": 512,
                    "min": 50,
                    "max": 2048,
                    "step": 50
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1
                }),

            },
            "optional": {
                "model_name": (cls.get_available_models(), {
                    "default": "Qwen3-VL-2B-Instruct"
                }),
                "device": (["Auto", "cuda", "cpu", "mps"], {
                    "default": "Auto"
                }),
                "quantization": (["无（FP16）", "4-bit", "8-bit"], {
                    "default": "4-bit"
                }),
                "attention_type": (["Eager: 最佳兼容性", "SDPA: 平衡", "Flash Attention 2: 最佳性能"], {
                    "default": "Eager: 最佳兼容性"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "description": "随机种子，-1表示随机"
                }),
                "clear_cache": ("BOOLEAN", {
                    "default": False
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("图像描述", "详细响应")
    FUNCTION = "generate_caption"
    CATEGORY = "QwenVL-ALL"
    
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
        # 获取基础模型列表，并检查本地是否已下载
        available_models = []
        models_dir = os.path.join(folder_paths.models_dir, "LLM", "Qwen-VL")
        
        # 添加基础模型，检查是否已下载
        for model_name in cls.BASE_MODEL_REPO_MAP.keys():
            # 检查本地是否存在该模型
            model_path = os.path.join(models_dir, model_name)
            if os.path.exists(model_path) and os.path.isdir(model_path):
                available_models.append(f"{model_name}（已下载）")
            else:
                available_models.append(model_name)
        
        # 扫描本地模型文件夹，添加用户自定义的本地模型
        if os.path.exists(models_dir):
            for item in os.listdir(models_dir):
                item_path = os.path.join(models_dir, item)
                # 检查是否为目录且不在基础模型列表中
                if os.path.isdir(item_path):
                    # 提取模型名称（去除"（已下载）"标记）
                    clean_model_name = item.replace("（已下载）", "")
                    if clean_model_name not in cls.BASE_MODEL_REPO_MAP.keys():
                        # 检查是否已经在列表中（避免重复）
                        display_name = f"{item}（已下载）" if "（已下载）" not in item else item
                        if display_name not in available_models:
                            available_models.append(display_name)
        
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
                # 检查是否为目录
                if os.path.isdir(item_path):
                    # 提取干净的模型名称（去除"（已下载）"标记）
                    clean_model_name = item.replace("（已下载）", "")
                    # 如果干净的模型名称不在映射中，则添加到映射中
                    if clean_model_name not in model_repo_map:
                        # 对于本地模型，使用目录名作为仓库ID
                        model_repo_map[clean_model_name] = clean_model_name
        
        return model_repo_map
    
    def get_device(self, device_preference):
        """获取设备"""
        if device_preference == "Auto":
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
        print(f"[Qwen3-VL Image Caption] 📥 开始下载模型: {model_id}")
        print(f"[Qwen3-VL Image Caption] 📂 保存路径: {local_dir}")
        print(f"[Qwen3-VL Image Caption] ⏳ 请耐心等待，下载可能需要几分钟到几十分钟...")
        print(f"[Qwen3-VL Image Caption] 💡 下载进度将在下方显示")
        print(f"{'='*70}\n")
        
        try:
            # snapshot_download 将自动为每个文件显示下载进度条
            snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            print(f"\n{'='*70}")
            print(f"[Qwen3-VL Image Caption] ✅ 模型下载完成: {model_id}")
            print(f"{'='*70}\n")
        except Exception as e:
            print(f"\n{'='*70}")
            print(f"[Qwen3-VL Image Caption] ❌ 模型下载失败: {e}")
            print(f"[Qwen3-VL Image Caption] 💡 解决方案:")
            print(f"[Qwen3-VL Image Caption]    1. 检查网络连接")
            print(f"[Qwen3-VL Image Caption]    2. 使用镜像站: export HF_ENDPOINT=https://hf-mirror.com")
            print(f"[Qwen3-VL Image Caption]    3. 使用代理: export HTTP_PROXY=http://127.0.0.1:7890")
            print(f"[Qwen3-VL Image Caption]    4. 手动下载模型到: {local_dir}")
            print(f"{'='*70}\n")
            raise
    
    def load_model(self, model_name, device, quantization="无（FP16）", attention_type="Eager: 最佳兼容性"):
        """加载模型"""
        # 提取干净的模型名称（去除"（已下载）"标记）
        clean_model_name = model_name.replace("（已下载）", "")
        
        if self.current_model_name == clean_model_name and self.model is not None:
            return
            
        # 清理之前的模型
        self.cleanup_model()
        
        # 获取模型映射
        model_repo_map = self.get_model_repo_map()
        
        # 从映射中获取仓库ID，使用干净的模型名称
        repo_id = model_repo_map.get(clean_model_name)
        if not repo_id:
            raise ValueError(f"不支持的模型: {clean_model_name}")
        
        # 使用干净的模型名称作为本地目录名
        model_checkpoint = os.path.join(
            folder_paths.models_dir, "LLM", "Qwen-VL", clean_model_name
        )
        
        # 对于HuggingFace模型（包含"/"），需要下载
        if "/" in repo_id:
            # 如果模型不存在则下载
            if not os.path.exists(model_checkpoint):
                self._download_model_with_progress(repo_id, model_checkpoint)
        else:
            # 对于本地模型，检查是否存在
            if not os.path.exists(model_checkpoint):
                raise ValueError(f"本地模型不存在: {model_checkpoint}")
        
        try:
            print(f"[Qwen3-VL Image Caption] 正在加载模型: {model_name} 到设备: {device}")
            
            # 设置模型加载参数
            model_kwargs = {
                "dtype": torch.float16 if device == "cuda" else torch.float32,
                "device_map": "auto" if device == "cuda" else None,
                "low_cpu_mem_usage": True,
                "trust_remote_code": True
            }
            
            # 配置量化
            quantization_config = None
            if quantization == "4-bit":
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
                model_kwargs["quantization_config"] = quantization_config
            elif quantization == "8-bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                model_kwargs["quantization_config"] = quantization_config
            
            # 配置注意力类型
            attention_type_map = {
                "Eager: 最佳兼容性": "eager",
                "SDPA: 平衡": "sdpa", 
                "Flash Attention 2: 最佳性能": "flash_attention_2"
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
            print(f"[Qwen3-VL Image Caption] 模型加载成功: {model_name}")
            
        except Exception as e:
            print(f"[Qwen3-VL Image Caption] 模型加载失败: {str(e)}")
            raise RuntimeError(f"模型加载失败: {str(e)}。请确保网络连接或手动将模型下载到 ComfyUI/models/LLM/Qwen-VL/ 目录")
    
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
    
    def tensor_to_pil(self, image_tensor):
        """将Tensor转换为PIL图像"""
        tensor = image_tensor
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
    
    def generate_caption(self, image, caption_prompt, preset_selection, system_prompt, max_tokens, temperature, seed=-1, model_name="Qwen3-VL-2B-Instruct", device="Auto", quantization="无（FP16）", attention_type="Eager: 最佳兼容性", clear_cache=False):
        """生成图像描述"""
        
        # 提取干净的模型名称（去除"（已下载）"标记）
        clean_model_name = model_name.replace("（已下载）", "")
        
        # 如果 caption_prompt 为空且选择了预设，则使用预设提示
        if not caption_prompt.strip() and preset_selection != "无预设" and preset_selection in self.CAPTION_PRESETS:
            caption_prompt = self.CAPTION_PRESETS[preset_selection]
        
        print(f"[Qwen3-VL] 调试信息 - 输入参数:")
        print(f"  - 清理缓存: {clear_cache}")
        print(f"  - 当前缓存状态: Model={self.model is not None}, Processor={self.processor is not None}")
        
        if clear_cache:
            self.cleanup_model()

        
        try:
            # 独立加载模式 - 使用原始加载方法
            print(f"[Qwen3-VL] 使用独立加载模式: {clean_model_name}")
            try:
                device_actual = self.get_device(device)
                self.load_model(clean_model_name, device_actual, quantization, attention_type)
                current_model = self.model
                current_processor = self.processor
                print(f"[Qwen3-VL Image Caption] 模型加载成功")
            except Exception as load_error:
                print(f"[Qwen3-VL Image Caption] 模型加载失败: {str(load_error)}")
                raise RuntimeError(f"模型加载失败: {str(load_error)}。请确保网络连接或手动将模型下载到 ComfyUI/models/LLM/Qwen-VL/ 目录")
            
            # 转换图像
            pil_image = self.tensor_to_pil(image)
            
            # 准备消息
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": caption_prompt}
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
            
            # 移动到适当的设备
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # 生成描述
            print(f"[Qwen3-VL Image Caption] 开始生成图像描述...")
            
            # 如果指定了种子则设置种子
            if seed != -1:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
            
            with torch.no_grad():
                generated_ids = current_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
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
            
            print(f"[Qwen3-VL Image Caption] 图像描述生成完成")
            
            # 提取纯描述文本（移除思考过程标签）
            import re
            clean_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
            
            return (clean_response, response_text)
            
        except Exception as e:
            error_msg = f"Image caption generation failed: {str(e)}"
            print(f"[Qwen3-VL Image Caption] {error_msg}")
            return (error_msg, f"Error details: {str(e)}")


# Node mappings
NODE_CLASS_MAPPINGS = {
    "Qwen3VLImageCaption": Qwen3VLImageCaption,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3VLImageCaption": "Qwen3-VL 图像反推",
}