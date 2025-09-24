"""
LLM Interface for the Digital Clone AI Assistant.

This module provides a unified interface for different LLM providers,
with support for function calling, streaming, and fallback mechanisms.
"""

import json
import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Generator, Union

import requests

# Conditional imports to support both relative and absolute imports
try:
    from .config import config
    from .message_types import Message, LLMResponse, StreamingChunk, ToolCall, Role
except ImportError:
    from config import config
    from message_types import Message, LLMResponse, StreamingChunk, ToolCall, Role

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    def __init__(self, model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or config.DEEPSEEK_API_KEY
        self.base_url = base_url or config.DEEPSEEK_BASE_URL
        # Set timeout based on model type
        if "reasoner" in model_name.lower():
            self.timeout = config.TIMEOUT_SECONDS_REASONER
        else:
            self.timeout = config.TIMEOUT_SECONDS_CHAT

    @abstractmethod
    def generate(
        self,
        messages: List[Message],
        functions: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        system_prompt: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[LLMResponse, Generator[StreamingChunk, None, None]]:
        """Generate a response from the LLM."""
        pass

    def _convert_messages_to_api_format(
        self,
        messages: List[Message],
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Convert internal Message objects to API format."""
        api_messages = []

        # Add system prompt if provided
        if system_prompt:
            api_messages.append({
                "role": "system",
                "content": system_prompt
            })

        for msg in messages:
            api_msg = {
                "role": msg.role.value,
                "content": msg.content
            }

            # Add function call information for assistant messages
            if msg.tool_call:
                api_msg["function_call"] = {
                    "name": msg.tool_call.name,
                    "arguments": json.dumps(msg.tool_call.arguments)
                }

            # Add function result for tool messages
            if msg.tool_result:
                api_msg["role"] = "function"
                api_msg["name"] = msg.tool_result.name
                api_msg["content"] = msg.tool_result.content

            api_messages.append(api_msg)

        return api_messages

    def _parse_api_response(self, response_data: Dict[str, Any]) -> LLMResponse:
        """Parse API response into LLMResponse object."""
        content = ""
        tool_calls = []

        # Extract content and function calls
        if "choices" in response_data and response_data["choices"]:
            choice = response_data["choices"][0]
            message = choice.get("message", {})

            content = message.get("content", "")

            # Handle function calls
            if "function_call" in message:
                func_call = message["function_call"]
                try:
                    arguments = json.loads(func_call.get("arguments", "{}"))
                except json.JSONDecodeError:
                    arguments = {}

                tool_calls.append(ToolCall(
                    name=func_call.get("name", ""),
                    arguments=arguments
                ))

            # Handle finish reason
            finish_reason = choice.get("finish_reason")

        usage = response_data.get("usage")

        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage
        )

    def _parse_streaming_chunk(self, chunk_data: Dict[str, Any]) -> StreamingChunk:
        """Parse a streaming chunk from API response."""
        content = ""
        tool_calls = []

        if "choices" in chunk_data and chunk_data["choices"]:
            choice = chunk_data["choices"][0]
            delta = choice.get("delta", {})

            content = delta.get("content") or ""

            # Handle function calls in streaming
            if "function_call" in delta:
                func_call = delta["function_call"]
                try:
                    arguments = json.loads(func_call.get("arguments", "{}"))
                except json.JSONDecodeError:
                    arguments = {}

                tool_calls.append(ToolCall(
                    name=func_call.get("name", ""),
                    arguments=arguments
                ))

        finish_reason = None
        if "choices" in chunk_data and chunk_data["choices"]:
            choice = chunk_data["choices"][0]
            finish_reason = choice.get("finish_reason")

        usage = chunk_data.get("usage")

        return StreamingChunk(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage
        )


class DeepSeekChatClient(LLMClient):
    """DeepSeek Chat model client."""

    def __init__(self):
        super().__init__(
            model_name=config.MODEL_CHAT,
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL
        )

    def generate(
        self,
        messages: List[Message],
        functions: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        system_prompt: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[LLMResponse, Generator[StreamingChunk, None, None]]:
        """Generate response using DeepSeek Chat model."""
        if not self.api_key:
            raise ValueError("DeepSeek API key is required")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": self._convert_messages_to_api_format(messages, system_prompt),
            "temperature": 0.7,
            "max_tokens": 2000,
            "stream": stream
        }

        if functions:
            payload["functions"] = functions

        if response_format:
            payload["response_format"] = response_format

        try:
            logger.debug(f"Sending request to {url} with model {self.model_name}")
            start_time = time.time()

            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
                stream=stream  # Enable streaming for response
            )

            duration = time.time() - start_time
            logger.debug(f"API call completed in {duration:.2f}s")

            response.raise_for_status()

            if stream:
                # Handle streaming response
                return self._handle_streaming_response(response)
            else:
                # Handle regular response
                response_data = response.json()
                return self._parse_api_response(response_data)

        except requests.Timeout:
            logger.error("DeepSeek Chat API request timed out")
            raise
        except requests.HTTPError as e:
            logger.error(f"DeepSeek Chat API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in DeepSeek Chat client: {e}")
            raise

    def _handle_streaming_response(self, response) -> Generator[StreamingChunk, None, None]:
        """Handle streaming response from DeepSeek API."""
        try:
            for line in response.iter_lines():
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    if data.strip() == '[DONE]':
                        break

                    try:
                        chunk_data = json.loads(data)
                        chunk = self._parse_streaming_chunk(chunk_data)
                        yield chunk
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error processing streaming response: {e}")
            raise


class DeepSeekReasonerClient(LLMClient):
    """DeepSeek Reasoner model client for complex tasks."""

    def __init__(self):
        super().__init__(
            model_name=config.MODEL_REASONER,
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.DEEPSEEK_BASE_URL
        )

    def generate(
        self,
        messages: List[Message],
        functions: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        system_prompt: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[LLMResponse, Generator[StreamingChunk, None, None]]:
        """Generate response using DeepSeek Reasoner model."""
        if not self.api_key:
            raise ValueError("DeepSeek API key is required")

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "messages": self._convert_messages_to_api_format(messages, system_prompt),
            "temperature": 0.1,  # Lower temperature for reasoning tasks
            "max_tokens": 4000,  # Higher token limit for complex reasoning
            "stream": stream
        }

        if functions:
            payload["functions"] = functions

        if response_format:
            payload["response_format"] = response_format

        try:
            logger.debug(f"Sending request to {url} with model {self.model_name}")
            start_time = time.time()

            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
                stream=stream  # Enable streaming for response
            )

            duration = time.time() - start_time
            logger.debug(f"API call completed in {duration:.2f}s")

            response.raise_for_status()

            if stream:
                # Handle streaming response
                return self._handle_streaming_response(response)
            else:
                # Handle regular response
                response_data = response.json()
                return self._parse_api_response(response_data)

        except requests.Timeout:
            logger.error("DeepSeek Reasoner API request timed out")
            raise
        except requests.HTTPError as e:
            logger.error(f"DeepSeek Reasoner API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in DeepSeek Reasoner client: {e}")
            raise

    def _handle_streaming_response(self, response) -> Generator[StreamingChunk, None, None]:
        """Handle streaming response from DeepSeek API."""
        try:
            for line in response.iter_lines():
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    if data.strip() == '[DONE]':
                        break

                    try:
                        chunk_data = json.loads(data)
                        chunk = self._parse_streaming_chunk(chunk_data)
                        yield chunk
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"Error processing streaming response: {e}")
            raise


class MockClient(LLMClient):
    """Mock client for development and testing without API keys."""

    def __init__(self):
        super().__init__(model_name="mock-client")

    def generate(
        self,
        messages: List[Message],
        functions: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
        system_prompt: Optional[str] = None,
        response_format: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Union[LLMResponse, Generator[StreamingChunk, None, None]]:
        """Generate mock response for development."""
        logger.info("Using MockClient for response generation")

        # Get the last user message
        last_user_msg = None
        for msg in reversed(messages):
            if msg.role == Role.USER:
                last_user_msg = msg
                break

        if not last_user_msg:
            response_content = "我没有收到您的消息，请重新输入。"
        else:
            user_content = last_user_msg.content.lower()

            # Handle JSON mode for planner
            if response_format and response_format.get("type") == "json_object":
                # Return mock JSON plan for competitive analysis
                response_content = '''{
  "goal": "进行公司竞品两周调研并制定方案",
  "success_criteria": "完成竞品技术优势分析，输出可执行的竞争策略建议",
  "todos": [
    {
      "id": "T1",
      "title": "调研竞品技术特点",
      "why": "了解市场主要竞争对手的技术优势",
      "type": "tool",
      "tool": "rag_search",
      "input": {"query": "AI助手竞品技术分析", "k": 3},
      "expected_output": "竞品技术特点总结",
      "needs": []
    },
    {
      "id": "T2",
      "title": "分析当前时间",
      "why": "确定调研的时间节点",
      "type": "tool",
      "tool": "datetime",
      "input": {},
      "expected_output": "当前日期时间",
      "needs": []
    },
    {
      "id": "T3",
      "title": "制定竞争策略",
      "why": "基于调研结果制定应对策略",
      "type": "reason",
      "input": {},
      "expected_output": "竞争策略建议",
      "needs": []
    }
  ]
}'''
                tool_calls = []
            else:
                # Simple mock responses based on content
                if "算" in user_content or "计算" in user_content:
                    # Mock calculator tool call
                    response_content = "我来帮您计算这个问题。"
                    tool_calls = [ToolCall(
                        name="calculator",
                        arguments={"expression": "2 + 3"}  # Mock expression
                    )]
                elif "时间" in user_content or "日期" in user_content:
                    # Mock datetime tool call
                    response_content = "我来告诉您当前的时间。"
                    tool_calls = [ToolCall(
                        name="datetime",
                        arguments={"format": "iso"}
                    )]
                elif "计划" in user_content or "规划" in user_content:
                    # Mock ask_user for complex planning
                    response_content = "这是一个复杂的规划任务，我需要更多信息来为您制定最佳方案。请告诉我：\n1. 您的目标是什么？\n2. 时间限制是多久？\n3. 您有什么特殊要求吗？"
                    tool_calls = []
                else:
                    # General response
                    response_content = "这是MockClient的回复。在实际使用中，这里会是真实的AI回答。您的问题是：" + last_user_msg.content
                    tool_calls = []

        if stream:
            # Simulate streaming by yielding chunks
            return self._mock_streaming_response(response_content, tool_calls)
        else:
            return LLMResponse(
                content=response_content,
                tool_calls=tool_calls
            )

    def _mock_streaming_response(self, content: str, tool_calls: List[ToolCall]) -> Generator[StreamingChunk, None, None]:
        """Mock streaming response by yielding content in chunks."""
        # Yield tool calls first if any
        if tool_calls:
            for tool_call in tool_calls:
                yield StreamingChunk(
                    content="",
                    tool_calls=[tool_call]
                )

        # Yield content in word-sized chunks
        words = content.split()
        current_chunk = ""

        for word in words:
            current_chunk += word + " "
            if len(current_chunk) > 10:  # Yield chunk when it gets long enough
                yield StreamingChunk(content=current_chunk)
                current_chunk = ""

        # Yield remaining content
        if current_chunk:
            yield StreamingChunk(content=current_chunk)

        # Yield final chunk with finish_reason
        yield StreamingChunk(
            content="",
            finish_reason="stop"
        )


def create_llm_client(client_type: str = "auto") -> LLMClient:
    """
    Factory function to create appropriate LLM client.

    Args:
        client_type: Type of client to create ("chat", "reasoner", "mock", or "auto")

    Returns:
        Appropriate LLMClient instance
    """
    if client_type == "mock" or (client_type == "auto" and config.should_use_mock_client()):
        logger.info("Creating MockClient (no API key or explicitly requested)")
        return MockClient()
    elif client_type == "chat":
        logger.info("Creating DeepSeekChatClient")
        return DeepSeekChatClient()
    elif client_type == "reasoner":
        logger.info("Creating DeepSeekReasonerClient")
        return DeepSeekReasonerClient()
    elif client_type == "auto":
        # Auto-selection logic will be handled by router
        logger.info("Auto-selecting LLM client (will be determined by router)")
        return DeepSeekChatClient()  # Default to chat
    else:
        raise ValueError(f"Unknown client type: {client_type}")
