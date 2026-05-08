"""
环境变量加载工具模块
用于从 .env 文件中加载 API 密钥和 URL 配置
"""

import os
from pathlib import Path
from typing import Optional


def load_env_file(env_path: str = "D:\Doctor_agent\src\LLM\.env") -> dict:
    """
    从 .env 文件加载环境变量
    
    Args:
        env_path: .env 文件路径，默认为当前目录下的 .env
        
    Returns:
        包含环境变量的字典
    """
    env_vars = {}
    env_file = Path(env_path)
    
    if not env_file.exists():
        raise FileNotFoundError(f"未找到环境变量文件: {env_path}")
    
    with open(env_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 跳过空行和注释
            if not line or line.startswith('#'):
                continue
            
            # 解析键值对
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()
                env_vars[key] = value
                # 同时设置到系统环境变量
                os.environ[key] = value
    
    return env_vars


def get_llm_config() -> dict:
    """
    获取 DeepSeek API 配置
    
    Returns:
        包含 api_key 和 base_url 的字典
    """
    deepseek_api_key = os.getenv('DEEPSEEK_APIKEY')
    deepseek_base_url = os.getenv('DEEPSEEK_BASEURL')

    dashscope_api_key = os.getenv('DASHSCOPE_APIKEY')
    dashscope_base_url = os.getenv('DASHSCOPE_BASEURL')
    if not deepseek_api_key:
        raise ValueError("未找到 DEEPSEEK_APIKEY 环境变量")
    if not deepseek_base_url:
        raise ValueError("未找到 DEEPSEEK_BASEURL 环境变量")

    if not dashscope_api_key:
        raise ValueError("未找到 DASHSCOPE_APIKEY 环境变量")
    if not dashscope_base_url:
        raise ValueError("未找到 DASHSCOPE_BASEURL 环境变量")
    
    return {
        'deepseek_api_key': deepseek_api_key,
        'deepseek_base_url':  deepseek_base_url,
        'dashscope_api_key': dashscope_api_key,
        'dashscope_base_url': dashscope_base_url
    }


def get_env_var(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    获取指定的环境变量
    
    Args:
        key: 环境变量名称
        default: 默认值，如果环境变量不存在则返回此值
        
    Returns:
        环境变量的值或默认值
    """
    return os.getenv(key, default)


# 自动加载 .env 文件
if __name__ != "__main__":
    try:
        load_env_file()
    except FileNotFoundError:
        pass  # 如果 .env 文件不存在，静默跳过


if __name__ == "__main__":
    # 测试代码
    print("正在加载环境变量...")
    env_vars = load_env_file()
    print(f"成功加载 {len(env_vars)} 个环境变量")
    
    print("\nDeepSeek 配置:")
    config = get_llm_config()
    print(config['deepseek_api_key'])
    print(config['deepseek_base_url'])
    print(config['dashscope_api_key'])
    print(config['dashscope_base_url'])
