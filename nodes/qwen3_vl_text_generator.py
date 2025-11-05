"""
Qwen3-VL 文本生成器节点
用于基于文本提示生成创意内容
"""

import torch
import json
import os
import folder_paths
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoProcessor
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    Qwen3VLForConditionalGeneration = None
from huggingface_hub import snapshot_download
import random
import re


class Qwen3VLTextGenerator:
    """Qwen3-VL 文本生成器 - 支持创意文本生成和内容创作"""
    
    # 模型ID映射：模型名称 -> HuggingFace仓库ID
    MODEL_REPO_MAP = {
        "Qwen3-VL-2B-Instruct": "Qwen/Qwen3-VL-2B-Instruct",
        "Qwen3-VL-4B-Instruct": "Qwen/Qwen3-VL-4B-Instruct"
    }
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "输入文本": ("STRING", {"default": "请帮我写一段关于人工智能的介绍", "multiline": True}),
                "生成模式": (["创意写作", "技术文档", "营销文案", "故事创作", "诗歌生成", "代码注释"],),
                "风格模板": (["正式", "轻松", "幽默", "专业", "感性", "理性"],),
                "最大长度": ("INT", {"default": 1024, "min": 100, "max": 4096, "step": 100}),
                "温度": ("FLOAT", {"default": 0.8, "min": 0.1, "max": 2.0, "step": 0.1}),
                "重复惩罚": ("FLOAT", {"default": 1.1, "min": 1.0, "max": 2.0, "step": 0.1}),

            },
            "optional": {
                "清理模型缓存": ("BOOLEAN", {"default": False, "description": "清理模型缓存，重新加载模型"}),
                "上下文信息": ("STRING", {"default": "", "multiline": True}),
                "关键词": ("STRING", {"default": "", "multiline": False}),
                "输出格式": ("STRING", {"default": "", "multiline": False}),
                "模型名称": (list(cls.MODEL_REPO_MAP.keys()), {"default": "Qwen3-VL-2B-Instruct"}),
                "设备": (["自动", "cuda", "cpu", "mps"], {"default": "自动"}),
                "量化": (["无", "4位量化", "8位量化"], {"default": "4位量化"}),
                "注意力类型": (["标准注意力：兼容性好", "SDPA注意力：平衡", "Flash注意力2：性能优先"], {"default": "标准注意力：兼容性好"}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("生成文本", "处理提示词", "生成统计", "调试信息")
    FUNCTION = "generate_text"
    CATEGORY = "QwenVL-ALL"
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.current_model_name = None
    
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
            print(f"[Qwen3-VL文本生成] 正在下载模型 {model_name}...")
            model_path = os.path.join(folder_paths.models_dir, "LLM", "Qwen-VL", model_name)
            
            snapshot_download(
                repo_id=repo_id,
                local_dir=model_path,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            print(f"[Qwen3-VL文本生成] 模型下载完成: {model_path}")
            return model_path
            
        except Exception as e:
            print(f"[Qwen3-VL文本生成] 模型下载失败: {str(e)}")
            print("[Qwen3-VL文本生成] 解决方法建议:")
            print("1. 检查网络连接")
            print("2. 确认HuggingFace访问权限")
            print("3. 手动下载模型到指定路径")
            raise e
        self.current_device = None
        self.current_attention_type = None
        self.current_quantization = None
    
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
                print(f"[Qwen3-VL文本生成] 使用缓存模型: {model_name}")
                return self.model, self.tokenizer
            
            # 清理之前的模型缓存
            if self.model is not None:
                print(f"[Qwen3-VL文本生成] 清理之前的模型缓存...")
                del self.model
                del self.tokenizer
                torch.cuda.empty_cache()
            
            # 获取模型仓库ID
            if model_name not in self.MODEL_REPO_MAP:
                raise ValueError(f"[Qwen3-VL文本生成] 不支持的模型名称: {model_name}")
            
            repo_id = self.MODEL_REPO_MAP[model_name]
            
            # 构建本地模型路径
            model_checkpoint = os.path.join(folder_paths.models_dir, "LLM", "Qwen-VL", model_name)
            
            # 检查模型是否已存在
            if not os.path.exists(model_checkpoint):
                print(f"[Qwen3-VL文本生成] 本地模型不存在，开始下载: {model_name}")
                model_checkpoint = self._download_model_with_progress(model_name, repo_id)
            else:
                print(f"[Qwen3-VL文本生成] 使用本地模型: {model_checkpoint}")
            
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
            
            print(f"[Qwen3-VL文本生成] 正在加载模型: {model_name}")
            print(f"[Qwen3-VL文本生成] 设备: {device_name}")
            print(f"[Qwen3-VL文本生成] 量化: {quantization}")
            print(f"[Qwen3-VL文本生成] 注意力类型: {attention_type}")
            
            # 加载处理器（包含分词器）
            self.processor = AutoProcessor.from_pretrained(
                model_checkpoint,
                trust_remote_code=True
            )
            self.tokenizer = self.processor.tokenizer
            
            # 检查是否支持Qwen3VL模型
            if Qwen3VLForConditionalGeneration is None:
                raise ImportError("当前transformers版本不支持Qwen3VL模型，请升级transformers到最新版本")
            
            # 加载模型 - 使用Qwen3VL专用模型类
            model_kwargs = {
                "trust_remote_code": True,
                "dtype": torch.float16 if quantization == "无" else "auto",
                "device_map": "auto" if quantization != "无" else None,
                "attn_implementation": attention_impl
            }
            
            if quantization_config is not None:
                model_kwargs["quantization_config"] = quantization_config
            
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_checkpoint,
                **model_kwargs
            )
            
            # 设置模型到指定设备（非量化模式）
            if quantization == "无" and device_name != "auto":
                self.model = self.model.to(device_name)
            
            # 更新当前状态
            self.current_model_name = model_name
            self.current_device = device_name
            self.current_attention_type = attention_type
            self.current_quantization = quantization
            
            print(f"[Qwen3-VL文本生成] 模型加载完成: {model_name}")
            return self.model, self.tokenizer
            
        except Exception as e:
            print(f"[Qwen3-VL文本生成] 模型加载失败: {str(e)}")
            print("[Qwen3-VL文本生成] 解决方法建议:")
            print("1. 检查模型文件是否完整")
            print("2. 确认模型路径配置正确")
            print("3. 检查设备内存是否充足")
            print("4. 尝试使用量化模式或切换注意力类型")
            raise e
    
    def build_prompt(self, 输入文本, 生成模式, 风格模板, 上下文信息, 关键词, 输出格式):
        """构建优化的提示词"""
        
        # 模式特定的系统提示词
        mode_prompts = {
            "创意写作": "你是一个创意写作助手，擅长生成富有想象力和创造力的文本内容。",
            "技术文档": "你是一个技术文档专家，能够生成准确、清晰的技术说明和文档。",
            "营销文案": "你是一个营销文案专家，擅长创作吸引人、有说服力的营销内容。",
            "故事创作": "你是一个故事创作大师，能够创作引人入胜的故事情节和人物对话。",
            "诗歌生成": "你是一个诗人，擅长创作优美、富有意境的诗歌作品。",
            "代码注释": "你是一个编程专家，能够为代码提供清晰、有用的注释和说明。"
        }
        
        # 风格特定的要求
        style_requirements = {
            "正式": "请使用正式、专业的语言风格。",
            "轻松": "请使用轻松、易懂的语言风格。",
            "幽默": "请使用幽默、风趣的语言风格。",
            "专业": "请使用专业、权威的语言风格。",
            "感性": "请使用感性、富有情感的语言风格。",
            "理性": "请使用理性、逻辑清晰的语言风格。"
        }
        
        # 构建系统提示词
        system_prompt = f"""{mode_prompts.get(生成模式, "你是一个智能文本生成助手。")}
{style_requirements.get(风格模板, "请根据内容需求调整语言风格。")}"""
        
        # 添加上下文信息
        if 上下文信息:
            system_prompt += f"\n\n背景信息：{上下文信息}"
        
        # 添加关键词要求
        if 关键词:
            system_prompt += f"\n\n必须包含的关键词：{关键词}"
        
        # 添加输出格式要求
        if 输出格式:
            system_prompt += f"\n\n输出格式要求：{输出格式}"
        
        # 构建完整的对话提示
        prompt = f"""<|im_start|>system
{system_prompt}
<|im_end|>
<|im_start|>user
{输入文本}
<|im_end|>
<|im_start|>assistant"""
        
        return prompt
    
    def generate_text(self, 输入文本, 生成模式, 风格模板, 最大长度, 温度, 重复惩罚,
                      清理模型缓存=False, 上下文信息="", 关键词="", 输出格式="",
                      模型名称="Qwen3-VL-2B-Instruct", 设备="自动", 量化="4位量化", 注意力类型="标准注意力：兼容性好"):
        """基于输入文本生成创意内容"""
        
             # 清理模型缓存（如果需要）
        if 清理模型缓存:
            print("[Qwen3-VL-文本生成器] 清理模型缓存")
            self.cleanup_model()
        
        # 加载模型
        current_model, current_tokenizer = self.load_model(模型名称, 设备, 注意力类型, 量化)
        
        # 构建优化的提示词
        prompt = self.build_prompt(输入文本, 生成模式, 风格模板, 上下文信息, 关键词, 输出格式)
        
        # 准备消息格式（Qwen3VL格式）
        messages = [
            {"role": "system", "content": "你是一个智能文本生成助手，能够根据用户需求生成高质量的文本内容。"},
            {"role": "user", "content": prompt}
        ]
        
        # 使用处理器处理文本输入
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # 编码输入
        inputs = self.tokenizer(text, return_tensors="pt", return_attention_mask=True)
        
        # 获取设备信息
        device = next(current_model.parameters()).device
        
        if device != "auto":
            inputs = inputs.to(device)
        
        # 生成文本
        with torch.no_grad():
            outputs = current_model.generate(
                **inputs,
                max_new_tokens=最大长度,
                temperature=温度,
                repetition_penalty=重复惩罚,
                do_sample=温度 > 0,
                top_p=0.9,
                top_k=50,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码生成的文本
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # 提取助手回复部分
        assistant_text = generated_text[len(text):].strip()
        
        # 清理和格式化输出
        assistant_text = re.sub(r'<\|im_.*?\|>', '', assistant_text).strip()
        
        # 统计信息
        input_tokens = len(inputs['input_ids'][0])
        output_tokens = len(outputs[0]) - input_tokens
        total_tokens = len(outputs[0])
        
        使用统计 = f"""输入令牌数: {input_tokens}
输出令牌数: {output_tokens}
总令牌数: {total_tokens}
生成模式: {生成模式}
风格模板: {风格模板}"""
        
        生成信息 = f"""模型: {self.current_model_name}
温度: {温度}
重复惩罚: {重复惩罚}
最大长度: {最大长度}
生成时间: {torch.cuda.Event(enable_timing=True) if self.current_device != 'cpu' else 'N/A'}"""
        
        # 调试信息
        debug_info = f"""
模型: {模型名称}
设备: {设备}
量化: {量化}
注意力类型: {注意力类型}
输入长度: {len(输入文本)}
输出长度: {len(assistant_text)}
生成模式: {生成模式}
风格模板: {风格模板}
温度: {温度}
重复惩罚: {重复惩罚}
""".strip()
        
        print(f"[Qwen3-VL-文本生成器] 生成完成，输出长度: {len(assistant_text)} 字符")
        
        return (assistant_text, 使用统计, 生成信息, debug_info)
    
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
        
        print("[Qwen3-VL-文本生成器] 模型缓存已清理")


# 节点映射
NODE_CLASS_MAPPINGS = {
    "Qwen3VLTextGenerator": Qwen3VLTextGenerator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen3VLTextGenerator": "Qwen3-VL 文本生成器",
}