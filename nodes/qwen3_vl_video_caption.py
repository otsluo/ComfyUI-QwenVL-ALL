# -*- coding: utf-8 -*-
"""
Qwen3-VL 视频反推节点
基于Qwen3-VL模型实现视频内容的智能描述和分析
"""

import torch
import json
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Any, Optional
import folder_paths
import comfy.model_management as model_management
import io
import base64
import locale
import sys

# 设置默认编码
if sys.platform == 'win32':
    import _locale
    _locale._getdefaultlocale = (lambda *args: ['en_US', 'utf8'])
try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    QWEN3VL_AVAILABLE = True
except ImportError:
    print("[警告] Qwen3VLForConditionalGeneration 不可用，将使用备用模型加载方式")
    QWEN3VL_AVAILABLE = False
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import os
from pathlib import Path
try:
    from huggingface_hub import snapshot_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False


class Qwen3VLVideoCaption:
    """
    Qwen3-VL 视频反推节点
    支持视频关键帧提取、内容描述、动作分析等功能
    """
    
    # 模型ID映射：模型名称 -> HuggingFace仓库ID
    MODEL_REPO_MAP = {
        "Qwen3-VL-2B-Instruct": "Qwen/Qwen3-VL-2B-Instruct",
        "Qwen3-VL-4B-Instruct": "Qwen/Qwen3-VL-4B-Instruct"
    }
    
    def __init__(self):
        # 初始化模型缓存
        self.model_cache = {}
        self.processor_cache = {}
        self.current_model_name = None
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "video_frames": ("IMAGE", {
                    "description": "输入视频帧序列"
                }),
                "caption_mode": (["详细描述", "简洁描述", "动作分析", "场景分析", "情感分析", "自定义"], {
                    "default": "详细描述",
                    "description": "选择视频描述模式"
                }),
                "max_tokens": ("INT", {
                    "default": 512,
                    "min": 50,
                    "max": 2048,
                    "step": 50,
                    "description": "最大令牌数"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "description": "温度参数"
                })
            },
            "optional": {
                "model_name": (["Qwen3-VL-2B-Instruct", "Qwen3-VL-4B-Instruct"], 
                                 {"default": "Qwen3-VL-2B-Instruct",
                                  "description": "选择Qwen3-VL模型版本"}),
                "device": (["auto", "cpu", "cuda", "mps"], {
                    "default": "auto",
                    "description": "设备选择"
                }),
                "attention_type": (["标准注意力", "SDPA注意力", "Flash注意力2"], {
                    "default": "SDPA注意力",
                    "description": "注意力类型"
                }),
                "quantization": (["无", "4位量化", "8位量化"], {
                    "default": "4位量化",
                    "description": "量化模式"
                }),
                "custom_prompt": ("STRING", {
                    "default": "请详细描述这个视频的内容，包括场景、动作、情感等",
                    "multiline": True,
                    "description": "自定义提示词(仅在自定义模式下使用)"
                }),
                "frame_sampling": (["均匀采样", "关键帧提取", "全部帧"], {
                    "default": "均匀采样",
                    "description": "帧采样策略"
                }),
                "sample_rate": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 60,
                    "step": 1,
                    "description": "采样帧数"
                }),
                "output_format": (["纯文本", "JSON格式", "Markdown格式"], {
                    "default": "纯文本",
                    "description": "输出格式"
                }),
                "keep_model_loaded": ("BOOLEAN", {
                    "default": False,
                    "description": "保持模型加载状态(减少重复加载时间但占用更多内存)"
                })
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "DICT")
    RETURN_NAMES = ("视频描述", "详细响应", "处理信息")
    OUTPUT_NODE = True
    FUNCTION = "generate_video_caption"
    CATEGORY = "QwenVL-ALL"
    
    def _get_device_info(self, device: str) -> Dict[str, Any]:
        """获取设备信息"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        device_info = {
            "device": device,
            "device_name": torch.cuda.get_device_name() if device == "cuda" else device,
            "memory_free": torch.cuda.mem_get_info()[0] if device == "cuda" else 0,
            "memory_total": torch.cuda.mem_get_info()[1] if device == "cuda" else 0
        }
        
        return device_info
    
    def _download_model_with_progress(self, model_id: str, local_dir: str):
        """下载模型并显示进度"""
        print(f"\n{'='*70}")
        print(f"[Qwen3-VL 视频反推] 📥 开始下载模型: {model_id}")
        print(f"[Qwen3-VL 视频反推] 📂 保存路径: {local_dir}")
        print(f"[Qwen3-VL 视频反推] ⏳ 请耐心等待，下载可能需要几分钟到几十分钟...")
        print(f"[Qwen3-VL 视频反推] 💡 下载进度将在下方显示")
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
            print(f"[Qwen3-VL 视频反推] ✅ 模型下载完成: {model_id}")
            print(f"{'='*70}\n")
        except Exception as e:
            print(f"\n{'='*70}")
            print(f"[Qwen3-VL 视频反推] ❌ 模型下载失败: {e}")
            print(f"[Qwen3-VL 视频反推] 💡 解决方法:")
            print(f"[Qwen3-VL 视频反推]    1. 检查网络连接")
            print(f"[Qwen3-VL 视频反推]    2. 使用镜像站: export HF_ENDPOINT=https://hf-mirror.com")
            print(f"[Qwen3-VL 视频反推]    3. 使用代理: export HTTP_PROXY=http://127.0.0.1:7890")
            print(f"[Qwen3-VL 视频反推]    4. 手动下载模型到: {local_dir}")
            print(f"{'='*70}\n")
            raise
    
    def _process_attention_type(self, attention_type: str) -> str:
        """处理注意力类型"""
        attention_map = {
            "标准注意力": "eager",
            "SDPA注意力": "sdpa", 
            "Flash注意力2": "flash_attention_2"
        }
        
        for key, value in attention_map.items():
            if key in attention_type:
                return value
        
        return "sdpa"  # 默认返回SDPA
    
    def _load_model(self, model_name: str, device: str, quantization: str, attn_implementation: str):
        """加载模型和处理器，使用指定配置"""
        # 检查缓存
        if self.current_model_name == model_name and self.model_cache.get(model_name) is not None:
            return self.model_cache[model_name], self.processor_cache.get(model_name)
        
        # 从映射中获取HuggingFace仓库ID，回退到Qwen/{model_name}
        model_id = self.MODEL_REPO_MAP.get(model_name, f"Qwen/{model_name}")
        
        # 使用模型名称作为本地目录名称
        model_checkpoint = os.path.join(
            folder_paths.models_dir, "LLM", "Qwen-VL", model_name
        )
        
        # 如果模型不存在则下载
        if not os.path.exists(model_checkpoint) and HF_HUB_AVAILABLE:
            self._download_model_with_progress(model_id, model_checkpoint)
        elif not os.path.exists(model_checkpoint):
            print(f"[警告] 模型目录不存在且huggingface_hub不可用，将尝试直接加载: {model_name}")
            model_checkpoint = model_name  # 回退到直接加载
        
        try:
            # 加载处理器
            processor = AutoProcessor.from_pretrained(
                model_checkpoint,
                trust_remote_code=True,
                local_files_only=False  # 允许下载处理器文件
            )
            
            # 配置量化
            quantization_config = None
            if quantization == "4位量化":
                quantization_config = BitsAndBytesConfig(load_in_4bit=True)
            elif quantization == "8位量化":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            # 检测bf16支持
            bf16_support = (
                torch.cuda.is_available()
                and torch.cuda.get_device_capability(device if device != "auto" else torch.device("cuda"))[0] >= 8
            ) if device != "cpu" else False
            
            # 选择模型加载类
            if QWEN3VL_AVAILABLE:
                model_class = Qwen3VLForConditionalGeneration
            else:
                print("[警告] Qwen3VLForConditionalGeneration不可用，使用AutoModelForVision2Seq")
                model_class = AutoModelForVision2Seq
            
            # 加载模型
            model = model_class.from_pretrained(
                model_checkpoint,
                dtype=torch.bfloat16 if bf16_support else torch.float16,
                device_map="auto" if device == "auto" else None,
                attn_implementation=attn_implementation,
                quantization_config=quantization_config,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                local_files_only=False  # 允许下载模型文件
            )
            
            # 设备特定处理
            if device != "auto" and device != "cpu" and quantization == "无":
                model = model.to(device)
            
            # 缓存模型
            self.model_cache[model_name] = model
            self.processor_cache[model_name] = processor
            self.current_model_name = model_name
            
            return model, processor
            
        except Exception as e:
            error_msg = f"模型加载失败: {str(e)}"
            print(f"[Qwen3-VL Video Caption] {error_msg}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(error_msg)
        
        # 确保模型文件编码正确
        import locale
        encoding = locale.getpreferredencoding()
        print(f"[Qwen3-VL Video Caption] 系统编码: {encoding}")
    
    def generate_video_caption(self, video_frames, caption_mode, max_tokens, 
                             temperature, model_name="Qwen3-VL-2B-Instruct", device="auto", 
                             attention_type="SDPA注意力", quantization="4位量化",
                             custom_prompt=None, frame_sampling="均匀采样", sample_rate=8,
                             output_format="纯文本", keep_model_loaded=False) -> Tuple[str, str, Dict[str, Any]]:
        """
        生成视频描述
        
        Args:
            video_frames: 输入视频帧序列
            model_name: 模型名称
            caption_mode: 描述模式
            max_tokens: 最大令牌数
            temperature: 温度参数
            device: 设备选择
            attention_type: 注意力类型
            quantization: 量化模式
            custom_prompt: 自定义提示词
            frame_sampling: 帧采样策略
            sample_rate: 采样帧数
            output_format: 输出格式
            keep_model_loaded: 是否保持模型加载状态
            
        Returns:
            Tuple[str, str, Dict]: 视频描述、详细响应、处理信息
        """
        try:
            # 帧采样处理
            sampled_frames = self._sample_frames(video_frames, frame_sampling, sample_rate)
            
            # 构建提示词
            prompt = self._build_prompt(caption_mode, custom_prompt)
            
            # 处理设备信息
            device_info = self._get_device_info(device)
            
            # 处理注意力类型
            attn_implementation = self._process_attention_type(attention_type)
            
            # 独立加载模型
            print(f"[Qwen3-VL 视频反推] 独立加载模型: {model_name}")
            model, processor = self._load_model(
                model_name=model_name,
                device=device_info["device"],
                quantization=quantization,
                attn_implementation=attn_implementation
            )
            
            # 处理视频帧为base64编码
            image_data = []
            if sampled_frames.dim() == 4:  # (T, H, W, C)
                frames = (sampled_frames * 255).byte().cpu().numpy()
                for frame in frames:
                    # 转换张量到PIL图像
                    pil_image = Image.fromarray(frame)
                    
                    # 将PIL图像转换为base64
                    img_buffer = io.BytesIO()
                    pil_image.save(img_buffer, format='PNG')
                    img_buffer.seek(0)
                    img_b64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                    image_data.append(f"data:image/png;base64,{img_b64}")
            
            # 构建消息
            messages = self._prepare_messages(prompt, image_data, None)
            
            with torch.no_grad():
                # 应用聊天模板
                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # 处理视觉信息
                image_inputs, video_inputs, video_kwargs = process_vision_info(
                    messages,
                    return_video_kwargs=True
                )
                
                # 如果fps是序列则修复video_kwargs
                if video_kwargs and 'fps' in video_kwargs:
                    fps_value = video_kwargs['fps']
                    # 如果fps是序列，取第一个元素
                    if isinstance(fps_value, (list, tuple)):
                        video_kwargs['fps'] = fps_value[0] if fps_value else 24
                
                # 准备输入
                inputs = processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    **video_kwargs
                )
                
                # 获取设备信息
                if hasattr(model, 'device'):
                    device = model.device
                else:
                    device = next(model.parameters()).device
                
                inputs = inputs.to(device)
                
                # 生成
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=processor.tokenizer.eos_token_id,
                )
                
                generated_ids_trimmed = [
                    out_ids[len(in_ids):]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                result = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                
                output_text = result[0] if result else ""
            
            # 清理模型缓存（仅在不需要保持加载时）
            if not keep_model_loaded:
                if model_name in self.model_cache:
                    del self.model_cache[model_name]
                if model_name in self.processor_cache:
                    del self.processor_cache[model_name]
                self.current_model_name = None
                
                # 清理GPU内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            
            # 格式化输出
            formatted_output = self._format_output(output_text, output_format)
            
            # 构建处理信息
            process_info = {
                "model_name": model_name,
                "caption_mode": caption_mode,
                "frame_count": len(sampled_frames),
                "total_frames": len(video_frames),
                "device": str(device),
                "attention_type": attention_type,
                "quantization": quantization,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "processing_time": "N/A",  # 可以添加实际处理时间
                "output_length": len(output_text),
                "frame_sampling": frame_sampling,
                "sample_rate": sample_rate,
                "keep_model_loaded": keep_model_loaded
            }
            
            return (formatted_output, output_text, process_info)
            
        except Exception as e:
            error_msg = f"视频描述生成失败: {str(e)}"
            error_info = {
                "error": error_msg,
                "model_name": model_name,
                "caption_mode": caption_mode,
                "device": device,
                "frame_sampling": frame_sampling,
                "keep_model_loaded": keep_model_loaded
            }
            return (error_msg, error_msg, error_info)
    
    def _sample_frames(self, video_frames: torch.Tensor, strategy: str, rate: int) -> torch.Tensor:
        """帧采样处理"""
        total_frames = len(video_frames)
        
        if strategy == "全部帧":
            return video_frames
        elif strategy == "关键帧提取":
            # 简单的关键帧提取逻辑（可以改进为更复杂的算法）
            indices = torch.linspace(0, total_frames-1, min(rate, total_frames), dtype=torch.long)
            return video_frames[indices]
        else:  # 均匀采样
            indices = torch.linspace(0, total_frames-1, min(rate, total_frames), dtype=torch.long)
            return video_frames[indices]
    
    def _build_prompt(self, mode: str, custom_prompt: Optional[str] = None) -> str:
        """构建提示词"""
        prompts = {
            "详细描述": "请详细描述这个视频的内容，包括场景、人物、动作、情感、氛围等方面。",
            "简洁描述": "请简洁地描述这个视频的主要内容。",
            "动作分析": "请分析视频中的动作和行为，描述发生了什么。",
            "场景分析": "请描述视频中的场景、环境和背景。",
            "情感分析": "请分析视频中表达的情感和氛围。"
        }
        
        if mode == "自定义" and custom_prompt:
            return custom_prompt
        else:
            return prompts.get(mode, prompts["详细描述"])
    
    def _process_video_frames(self, frames: torch.Tensor, processor) -> List[Image.Image]:
        """处理视频帧"""
        processed_images = []
        
        for frame in frames:
            # 转换张量到PIL图像
            if isinstance(frame, torch.Tensor):
                # 假设帧是 [H, W, C] 格式，值范围 [0, 1]
                frame_np = (frame.cpu().numpy() * 255).astype(np.uint8)
                image = Image.fromarray(frame_np)
            else:
                image = frame
            
            processed_images.append(image)
        
        return processed_images
    
    def _build_messages(self, prompt: str, images: List[Image.Image]) -> List[Dict[str, Any]]:
        """构建对话消息"""
        content = [{"type": "text", "text": prompt}]
        
        # 添加图像
        for image in images:
            content.append({"type": "image", "image": image})
        
        messages = [
            {
                "role": "user",
                "content": content
            }
        ]
        
        return messages
    
    def _prepare_messages(self, text_prompt: str, image_data: Optional[List[str]], video_data: Optional[str]) -> List[Dict[str, Any]]:
        """准备消息内容，支持文本、图像和视频"""
        messages = []
        content = [{"type": "text", "text": text_prompt}]
        
        # 添加图像数据
        if image_data:
            for img_data in image_data:
                content.append({"type": "image", "image": img_data})
        
        # 添加视频数据
        if video_data:
            content.append({"type": "video", "video": video_data})
        
        messages.append({"role": "user", "content": content})
        return messages
    
    def _format_output(self, text: str, format_type: str) -> str:
        """格式化输出"""
        if format_type == "JSON格式":
            try:
                # 尝试将文本解析为JSON格式
                structured = {
                    "description": text,
                    "summary": text[:200] + "..." if len(text) > 200 else text,
                    "length": len(text)
                }
                return json.dumps(structured, ensure_ascii=False, indent=2)
            except:
                return text
        elif format_type == "Markdown格式":
            return f"## 视频描述\n\n{text}\n\n---\n*生成时间: {torch.cuda.Event().elapsed_time()}ms*"
        else:
            return text


# 节点映射
NODE_CLASS_MAPPINGS = {
    "Qwen3VLVideoCaption": Qwen3VLVideoCaption
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3VLVideoCaption": "Qwen3-VL 视频反推"
}