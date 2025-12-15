"""Enhanced LLM client using LiteLLM for multi-provider support."""

from typing import List, Optional, Dict, Any, Union
from litellm import completion, batch_completion
# 删除这些不存在的导入
# from litellm.types.utils import BaseMessage, ModelResponse  # 添加必要的导入

from ..config.logging_config import get_logger
from ..config.settings import LLMConfig


logger = get_logger("llm.client")


class LLMClient:
    """
    LLM client wrapper using LiteLLM for multi-provider support.
    
    LiteLLM automatically handles API keys from environment variables.
    Set the appropriate environment variable for your provider:
    
    Example usage:
    ```python
    import os
    from litellm import completion
    
    # Set your API key
    os.environ["OPENAI_API_KEY"] = "your-api-key"
    
    # Use model in format "provider/model" or configure provider separately
    response = completion(
        model="openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    ```
    
    Common environment variables:
    - OPENAI_API_KEY for OpenAI models
    - ANTHROPIC_API_KEY for Claude models  
    - GEMINI_API_KEY for Google Gemini
    - AZURE_API_KEY for Azure OpenAI
    - And many more - see LiteLLM documentation
    """
    
    def __init__(self, config: LLMConfig):
        """Initialize LLM client with configuration."""
        self.config = config
        self.model = config.model
        self.count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        
        logger.info(f"Initialized LLM client with model: {self.model}")
        logger.info("Ensure your API key is set as an environment variable for your LLM provider")
    
    def __repr__(self) -> str:
        """String representation of the client."""
        return (f'LLMClient(model={self.model}, calls={self.count}, '
                f'input_tokens={self.total_input_tokens}, '
                f'output_tokens={self.total_output_tokens}, '
                f'total_tokens={self.total_tokens})')
    
    def call(self, messages: Union[List[Dict[str, str]], str], **kwargs) -> str:  # 修正返回类型
        """
        Make a single LLM call.
        
        Args:
            messages: List of messages in the format [{"role": str, "content": str}] or a string
            **kwargs: Additional parameters to pass to the LLM API
            
        Returns:
            LLM response text
        """
        try:
            logger.debug(f"Making API call #{self.count + 1}")

            # 处理字符串输入，转换为消息格式
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            
            # 构建参数字典
            completion_kwargs = {
                "model": self.config.model,
                "messages": messages,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "enable_thinking": False,  # 添加这一行
            }
            
            # 如果api_base存在且不为空，则添加到参数中
            if self.config.api_base:
                completion_kwargs["api_base"] = self.config.api_base
                
            # 添加其他可能的kwargs
            completion_kwargs.update(kwargs)
            
            response = completion(**completion_kwargs)  # 修复引用问题
            
            # Track token usage
            usage = response.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
            
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_tokens += total_tokens
            self.count += 1
            
            logger.info(f"API Call #{self.count} - Input: {input_tokens}, "
                       f"Output: {output_tokens}, Total: {total_tokens} tokens")
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise

    def call_batch(self, messages_list: Union[List[List[Dict[str, str]]], List[str]], **kwargs) -> List[str]:  # 修正返回类型
        """
        Make batch LLM calls.
        
        Args:
            messages_list: List of message lists, each in the format [{"role": str, "content": str}] or list of strings
            **kwargs: Additional parameters to pass to the LLM API
            
        Returns:
            List of LLM response texts
        """
        try:
            logger.debug(f"Making batch API call with {len(messages_list)} prompts")
            
            # 处理字符串列表输入，转换为消息格式
            if messages_list and isinstance(messages_list[0], str):
                messages_list = [{"role": "user", "content": msg} for msg in messages_list]
            
            completion_kwargs = {
                "model": self.config.model,
                "messages": messages_list,
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
            }
            
            # 如果api_base存在且不为空，则添加到参数中
            if self.config.api_base:
                completion_kwargs["api_base"] = self.config.api_base
                
            completion_kwargs.update(kwargs)
            
            responses = batch_completion(**completion_kwargs)  # 修复引用问题
            
            # Track token usage for all responses
            total_batch_tokens = 0
            results = []
            
            for i, response in enumerate(responses):
                usage = response.usage
                input_tokens = usage.prompt_tokens
                output_tokens = usage.completion_tokens
                total_tokens = usage.total_tokens
                
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                self.total_tokens += total_tokens
                total_batch_tokens += total_tokens
                
                logger.info(f"Batch API Call #{self.count + i + 1} - Input: {input_tokens}, "
                           f"Output: {output_tokens}, Total: {total_tokens} tokens")
                
                results.append(response.choices[0].message.content.strip())
            
            self.count += len(messages_list)
            logger.info(f"Batch completed: {len(messages_list)} calls, {total_batch_tokens} total tokens")
            
            return results
            
        except Exception as e:
            logger.error(f"Batch LLM API call failed: {e}")
            raise
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            "model": self.model,
            "total_calls": self.count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "average_tokens_per_call": self.total_tokens / self.count if self.count > 0 else 0
        }
    
    def reset_stats(self) -> None:
        """Reset usage statistics."""
        self.count = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        logger.info("Usage statistics reset")


def create_llm_client(config: LLMConfig) -> LLMClient:
    """Factory function to create LLM client."""
    return LLMClient(config)

# 删除下面这个不属于任何类的to_dict函数
# def to_dict(self) -> Dict[str, Any]:
#     return {
#         "model": self.config.model,
#         "temperature": self.config.temperature,
#         "max_tokens": self.config.max_tokens,
#         "api_base": self.config.api_base,  # 添加这一行
#         # 删除下面这一行错误的语法
#         # LLM_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
#     }