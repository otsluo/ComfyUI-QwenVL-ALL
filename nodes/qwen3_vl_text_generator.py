"""
Qwen3-VL 文本生成器节点
用于基于文本提示生成创意内容
"""

import torch
import json
import os
import folder_paths
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoProcessor
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None
from huggingface_hub import snapshot_download
import random
import re


class Qwen3VLTextGenerator:
    """Qwen3-VL文本生成器 - 支持创意文本生成和内容创作"""
    
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
                # 检查是否为目录
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
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "写一个关于勇敢骑士和魔法龙的创意故事。",
                    "multiline": True,
                    "description": "输入提示词"
                }),
                "system_prompt": ("STRING", {
                    "default": "你是一个能够生成引人入胜故事和内容的创意作家。",
                    "multiline": True,
                    "description": "系统提示词，用于设定AI角色和行为"
                }),
                "max_tokens": ("INT", {
                    "default": 512,
                    "min": 50,
                    "max": 2048,
                    "step": 50,
                    "description": "要生成的最大令牌数"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "description": "生成温度参数，控制输出随机性"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "description": "核采样参数，控制输出多样性"
                }),
                "repetition_penalty": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 2.0,
                    "step": 0.1,
                    "description": "重复惩罚参数，控制重复内容的生成"
                }),
            },
            "optional": {
                "model_name": (cls.get_available_models(), {
                    "default": "Qwen3-VL-2B-Instruct",
                    "description": "选择Qwen3-VL模型版本"
                }),
                "device": (["Auto", "cuda", "cpu", "mps"], {
                    "default": "Auto",
                    "description": "设备选择"
                }),
                "quantization": (["无（FP16）", "4位", "8位"], {
                    "default": "4位",
                    "description": "模型量化选项"
                }),
                "attention_type": (["Eager: 最佳兼容性", "SDPA: 平衡", "Flash Attention 2: 最佳性能"], {
                    "default": "Eager: 最佳兼容性",
                    "description": "注意力机制类型"
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 0xffffffffffffffff,
                    "description": "随机种子，-1表示随机"
                }),
                "max_memory": (["无限制", "8GB", "10GB", "12GB", "16GB", "20GB", "24GB"], {
                    "default": "无限制",
                    "description": "限制模型在不同设备上的最大内存使用"
                }),
                "clear_cache": ("BOOLEAN", {
                    "default": False,
                    "description": "是否清除模型缓存"
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("生成文本", "详细响应", "使用统计", "调试信息")
    FUNCTION = "generate_text"
    CATEGORY = "QwenVL-ALL"
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.current_model_name = None
    
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
    
    def _parse_max_memory_option(self, max_memory_option):
        """解析max_memory选项为实际的内存配置"""
        memory_configs = {
            "无限制": {},
            "8GB": {"cuda:0": "8GiB", "cpu": "16GiB"},
            "10GB": {"cuda:0": "10GiB", "cpu": "20GiB"},
            "12GB": {"cuda:0": "12GiB", "cpu": "24GiB"},
            "16GB": {"cuda:0": "16GiB", "cpu": "32GiB"},
            "20GB": {"cuda:0": "20GiB", "cpu": "40GiB"},
            "24GB": {"cuda:0": "24GiB", "cpu": "48GiB"}
        }
        
        # 如果是预设选项，直接返回对应的配置
        if max_memory_option in memory_configs:
            config = memory_configs[max_memory_option]
            # 转换字符串格式的内存大小为字节数
            if config:
                parsed_config = {}
                import re
                for device_id, mem_str in config.items():
                    # 匹配数字和单位
                    match = re.match(r'^(\d+(?:\.\d+)?)([TGMK]iB|B)$', mem_str, re.IGNORECASE)
                    if match:
                        value, unit = match.groups()
                        value = float(value)
                        # 转换为字节
                        if unit.upper() == 'TB':
                            value *= 1024**4
                        elif unit.upper() == 'GB':
                            value *= 1024**3
                        elif unit.upper() == 'MB':
                            value *= 1024**2
                        elif unit.upper() == 'KB':
                            value *= 1024
                        elif unit.upper() == 'TIB':
                            value *= 1024**4
                        elif unit.upper() == 'GIB':
                            value *= 1024**3
                        elif unit.upper() == 'MIB':
                            value *= 1024**2
                        elif unit.upper() == 'KIB':
                            value *= 1024
                        parsed_config[device_id] = int(value)
                return parsed_config
            return config
        
        # 不支持自定义选项，返回无限制配置
        print(f"[Qwen3-VL 文本生成] 不支持的max_memory选项: {max_memory_option}，使用无限制配置")
        return {}
    
    def _download_model_with_progress(self, model_id: str, local_dir: str):
        """下载模型并显示进度"""
        print(f"\n{'='*70}")
        print(f"[Qwen3-VL 文本生成] 📥 开始下载模型: {model_id}")
        print(f"[Qwen3-VL 文本生成] 📂 保存路径: {local_dir}")
        print(f"[Qwen3-VL 文本生成] ⏳ 请耐心等待，下载可能需要几分钟到几十分钟...")
        print(f"[Qwen3-VL 文本生成] 💡 下载进度将在下方显示")
        print(f"{'='*70}\n")
        
        try:
            # snapshot_download 将自动显示每个文件的下载进度条
            snapshot_download(
                repo_id=model_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            print(f"\n{'='*70}")
            print(f"[Qwen3-VL 文本生成] ✅ 模型下载完成: {model_id}")
            print(f"{'='*70}\n")
        except Exception as e:
            print(f"\n{'='*70}")
            print(f"[Qwen3-VL 文本生成] ❌ 模型下载失败: {e}")
            print(f"[Qwen3-VL 文本生成] 💡 解决方案:")
            print(f"[Qwen3-VL 文本生成]    1. 检查网络连接")
            print(f"[Qwen3-VL 文本生成]    2. 使用镜像站: export HF_ENDPOINT=https://hf-mirror.com")
            print(f"[Qwen3-VL 文本生成]    3. 使用代理: export HTTP_PROXY=http://127.0.0.1:7890")
            print(f"[Qwen3-VL 文本生成]    4. 手动下载模型到: {local_dir}")
            print(f"{'='*70}\n")
            raise
    
    def load_model(self, model_name, device, attention_type, quantization, max_memory="无限制"):
        """加载模型，支持独立下载和缓存机制"""
        # 提取干净的模型名称（去除"（已下载）"标记）
        clean_model_name = model_name.replace("（已下载）", "")
        
        # 设备映射
        DEVICE_MAP = {
            "Auto": "auto",
            "CPU": "cpu",
            "GPU": "cuda",
            "MPS": "mps"
        }
        
        # 量化配置
        # 量化配置
        quantization_config = None
        if quantization == "4位":
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16
                )
            except Exception as e:
                print(f"[Qwen3-VL 文本生成] 4-bit量化配置失败: {str(e)}")
                print("[Qwen3-VL 文本生成] 回退到无量化")
                # 如果4-bit量化失败，回退到无量化
                quantization = "无（FP16）"
        elif quantization == "8位":
            try:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            except Exception as e:
                print(f"[Qwen3-VL 文本生成] 8-bit量化配置失败: {str(e)}")
                print("[Qwen3-VL 文本生成] 回退到无量化")
                quantization = "无（FP16）"
        
        # 注意力实现配置
        ATTN_IMPLEMENTATIONS = {
            "Eager: 最佳兼容性": "eager",
            "Flash Attention 2: 更快但兼容性较差": "flash_attention_2",
            "无": None
        }

        # 如果设备、注意力类型或量化发生变化则重新加载模型
        if (self.model is None or 
            self.current_device != device or 
            self.current_attention_type != attention_type or 
            self.current_quantization != quantization):
            
            # 清理旧模型
            if self.model is not None:
                del self.model
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            try:
                print(f"[Qwen3-VL 文本生成] 🔄 切换模型配置...")
                print(f"[Qwen3-VL 文本生成] 📱 当前设备: {device}")
                print(f"[Qwen3-VL 文本生成] 🔍 注意力机制: {attention_type}")
                print(f"[Qwen3-VL 文本生成] 📉 量化级别: {quantization}")
                
                # 获取模型映射
                model_repo_map = self.get_model_repo_map()
                
                # 从映射中获取仓库ID，使用干净的模型名称
                repo_id = model_repo_map.get(clean_model_name)
                if not repo_id:
                    raise ValueError(f"[Qwen3-VL 文本生成] 不支持的模型名称: {clean_model_name}")
                
                # 使用干净的模型名称作为本地目录名
                model_checkpoint = os.path.join(folder_paths.models_dir, "LLM", "Qwen-VL", clean_model_name)
                
                # 对于HuggingFace模型（包含"/"），需要下载
                if "/" in repo_id:
                    # 检查模型是否已存在
                    if not os.path.exists(model_checkpoint):
                        print(f"[Qwen3-VL 文本生成] 本地模型不存在，开始下载: {model_name}")
                        self._download_model_with_progress(repo_id, model_checkpoint)
                    else:
                        print(f"[Qwen3-VL 文本生成] 使用本地模型: {model_checkpoint}")
                else:
                    # 对于本地模型，检查是否存在
                    if not os.path.exists(model_checkpoint):
                        raise ValueError(f"[Qwen3-VL 文本生成] 本地模型不存在: {model_checkpoint}")
                
                # 获取实际设备
                actual_device = DEVICE_MAP.get(device, device)
                
                # 准备加载参数
                load_kwargs = {
                    "device_map": actual_device,
                    "attn_implementation": ATTN_IMPLEMENTATIONS.get(attention_type, attention_type),
                    "torch_dtype": torch.float16 if actual_device != "cpu" else torch.float32,
                    "trust_remote_code": True
                }
                
                # 配置max_memory
                max_memory_config = self._parse_max_memory_option(max_memory)
                if max_memory_config:
                    load_kwargs["max_memory"] = max_memory_config
                
                # 如需要则添加量化配置
                if quantization_config is not None:
                    load_kwargs["quantization_config"] = quantization_config
                
                print(f"[Qwen3-VL 文本生成] 🚀 加载模型...")
                self.processor = AutoProcessor.from_pretrained(model_checkpoint, trust_remote_code=True)
                if Qwen3VLForConditionalGeneration is not None:
                    self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_checkpoint, **load_kwargs)
                else:
                    raise RuntimeError("[Qwen3-VL 文本生成] ❌ 无法导入Qwen3VLForConditionalGeneration类，请检查transformers版本")
                self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, trust_remote_code=True)
                    
                # 更新当前配置
                self.current_model_name = model_name
                self.current_device = device
                self.current_attention_type = attention_type
                self.current_quantization = quantization
                
                print(f"[Qwen3-VL 文本生成] ✅ 模型加载成功!")
                
            except Exception as e:
                error_msg = f"[Qwen3-VL 文本生成] ❌ 模型加载失败: {str(e)}"
                print(error_msg)
                print("[Qwen3-VL 文本生成] 💡 解决方案:")
                print("1. 检查模型路径是否正确")
                print("2. 验证设备和量化配置是否支持")
                print("3. 查看上方详细错误信息")
                raise RuntimeError(error_msg)
    
    def build_prompt(self, prompt):
        """构建优化的提示词"""
        # 如果所有部分都为空，提供默认提示
        if not prompt:
            return "写一个关于勇敢骑士和魔法龙的创意故事。"
        
        return prompt

    def generate_text(self, prompt, system_prompt, max_tokens, temperature, top_p, 
                     repetition_penalty, seed, model_name, device, quantization="无（FP16）", 
                     attention_type="Eager: 最佳兼容性", clear_cache=False, max_memory="无限制"):
        """生成文本 - 支持多种创作模式和参数控制"""
        try:
            # 清理模型缓存（如需要）
            if clear_cache:
                self.cleanup_model()
                print("[Qwen3-VL 文本生成] 🧹 模型缓存已清理")
            
            # 加载模型
            print(f"[Qwen3-VL 文本生成] 🚀 准备加载模型: {model_name}")
            self.load_model(model_name, device, attention_type, quantization, max_memory)
            
            # 使用build_prompt方法处理参数
            processed_prompt = self.build_prompt(prompt)
            
            # 构建对话消息
            messages = []
            
            # 处理系统提示
            if system_prompt and system_prompt.strip():
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": processed_prompt})
            
            # 应用聊天模板
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # 编码输入
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
            
            # 生成参数
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "do_sample": True
            }
            
            print(f"[Qwen3-VL 文本生成] 📝 开始文本生成...")
            print(f"[Qwen3-VL 文本生成] 📥 输入提示词: {prompt[:100]}...")
            
            # 设置随机种子（如指定）
            if seed != -1:
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
            
            # 执行生成
            generated_ids = self.model.generate(
                **model_inputs,
                **generation_kwargs
            )
            
            # 处理输出
            generated_ids = generated_ids[:, model_inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            
            # 生成统计信息
            input_tokens = model_inputs['input_ids'].shape[1]
            output_tokens = generated_ids.shape[1]
            total_tokens = input_tokens + output_tokens
            
            stats = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "model_name": model_name,
                "device": device,
                "generation_params": {
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "repetition_penalty": repetition_penalty
                }
            }
            
            debug_info = {
                "prompt": prompt,
                "system_prompt": system_prompt,
                "full_response": response,
                "messages": messages,
                "model_inputs_shape": model_inputs['input_ids'].shape
            }
            
            print(f"[Qwen3-VL 文本生成] ✅ 文本生成完成!")
            print(f"[Qwen3-VL 文本生成] 📤 生成长度: {output_tokens} 个令牌")
            
            import json
            stats_json = json.dumps(stats, ensure_ascii=False, indent=2)
            debug_info_json = json.dumps(debug_info, ensure_ascii=False, indent=2)
            
            return (response, response, stats_json, debug_info_json)
            
        except Exception as e:
            error_msg = f"[Qwen3-VL 文本生成] ❌ 文本生成失败: {str(e)}"
            print(error_msg)
            print("[Qwen3-VL 文本生成] 💡 解决方案:")
            print("1. 检查模型是否正确加载")
            print("2. 验证输入参数是否合理")
            print("3. 查看上方详细错误信息")
            raise RuntimeError(error_msg)

    def cleanup_model(self):
        """清理模型缓存"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        self.current_model_name = None
        self.current_device = None
        self.current_attention_type = None
        self.current_quantization = None
        
        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("[Qwen3-VL 文本生成] 🧹 模型缓存已清理")


# 节点映射
NODE_CLASS_MAPPINGS = {
    "Qwen3VLTextGenerator": Qwen3VLTextGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3VLTextGenerator": "Qwen3-VL 文本生成器"
}