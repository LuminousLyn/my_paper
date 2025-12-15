import logging
import os
from pathlib import Path

def setup_logging():
    """设置日志配置"""
    # 创建日志目录
    log_dir = Path(__file__).parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # 主日志配置
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)
    
    # 代码生成专用日志配置
    code_gen_logger = logging.getLogger('code_generation')
    code_gen_logger.setLevel(logging.INFO)
    
    # 创建formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # 主日志文件handler
    main_handler = logging.FileHandler(log_dir / 'table_agent1.log', delay=False)
    main_handler.setFormatter(formatter)
    
    # 代码生成日志文件handler
    code_gen_handler = logging.FileHandler(log_dir / 'code_generation.log', delay=False)
    code_gen_handler.setFormatter(formatter)
    
    # 控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # 清除现有handler
    for handler in main_logger.handlers[:]:
        main_logger.removeHandler(handler)
    for handler in code_gen_logger.handlers[:]:
        code_gen_logger.removeHandler(handler)
    
    # 添加handler
    main_logger.addHandler(main_handler)
    main_logger.addHandler(console_handler)
    code_gen_logger.addHandler(code_gen_handler)
    code_gen_logger.addHandler(console_handler)
    
    # 设置传播属性
    main_logger.propagate = False
    code_gen_logger.propagate = False
    
    return main_logger, code_gen_logger


def log_code_generation(action_type, code_type, code_content, prompt=None, response=None):
    """
    代码生成专用日志函数
    参数:
    - action_type: 操作类型（如"SQL代码生成"、"Python代码执行"等）
    - code_type: 代码类型（如"SQL"、"Python"等）
    - code_content: 生成的代码内容
    - prompt: 使用的提示词（可选）
    - response: LLM响应（可选）
    """
    code_gen_logger = logging.getLogger('code_generation')
    
    # 构建日志消息
    message_parts = [f"{action_type} - {code_type}代码"]
    
    if code_content:
        message_parts.append(f"代码内容: {code_content}")
    
    if prompt:
        message_parts.append(f"提示词: {prompt}")
    
    if response:
        message_parts.append(f"LLM响应: {response}")
    
    log_message = " | ".join(message_parts)
    code_gen_logger.info(log_message)