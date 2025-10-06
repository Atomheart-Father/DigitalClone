"""
Backward compatibility module for the Digital Clone AI Assistant.

This module provides backward compatibility imports for the old backend structure
to ensure existing code continues to work during the migration period.
"""

# Import all modules from new locations
try:
    # Core modules
    from ..core.config import config
    from ..core.kernel.agent_core import AgentCore, AgentRouter, agent
    from ..core.telemetry.logger import ConversationLogger, conversation_logger
    
    # Schema modules
    from ..schemas.messages import (
        Message, Role, RouteDecision, ConversationContext,
        LLMResponse, StreamingChunk, ToolCall, ToolExecutionResult,
        ToolMeta, create_user_message, create_assistant_message,
        create_tool_message, create_ask_user_message, create_system_message
    )
    
    # Adapter modules
    from ..adapters.llm_api import (
        LLMClient, DeepSeekChatClient, DeepSeekReasonerClient, MockClient,
        create_llm_client
    )
    
    # Tool modules
    from ..tools.registry import ToolRegistry, registry
    
    # Application modules
    from ..apps.cli import CLIApp, main as cli_main
    
    # Re-export for backward compatibility
    __all__ = [
        'config', 'AgentCore', 'AgentRouter', 'agent',
        'ConversationLogger', 'conversation_logger',
        'Message', 'Role', 'RouteDecision', 'ConversationContext',
        'LLMResponse', 'StreamingChunk', 'ToolCall', 'ToolExecutionResult',
        'ToolMeta', 'create_user_message', 'create_assistant_message',
        'create_tool_message', 'create_ask_user_message', 'create_system_message',
        'LLMClient', 'DeepSeekChatClient', 'DeepSeekReasonerClient', 'MockClient',
        'create_llm_client', 'ToolRegistry', 'registry', 'CLIApp', 'cli_main'
    ]
    
except ImportError as e:
    # If new modules can't be imported, try to import from old locations
    try:
        from .config import config
        from .agent_core import AgentCore, AgentRouter, agent
        from .logger import ConversationLogger, conversation_logger
        from .message_types import (
            Message, Role, RouteDecision, ConversationContext,
            LLMResponse, StreamingChunk, ToolCall, ToolExecutionResult,
            ToolMeta, create_user_message, create_assistant_message,
            create_tool_message, create_ask_user_message, create_system_message
        )
        from .llm_interface import (
            LLMClient, DeepSeekChatClient, DeepSeekReasonerClient, MockClient,
            create_llm_client
        )
        from .tool_registry import ToolRegistry, registry
        from .cli_app import CLIApp
        
        # Re-export for backward compatibility
        __all__ = [
            'config', 'AgentCore', 'AgentRouter', 'agent',
            'ConversationLogger', 'conversation_logger',
            'Message', 'Role', 'RouteDecision', 'ConversationContext',
            'LLMResponse', 'StreamingChunk', 'ToolCall', 'ToolExecutionResult',
            'ToolMeta', 'create_user_message', 'create_assistant_message',
            'create_tool_message', 'create_ask_user_message', 'create_system_message',
            'LLMClient', 'DeepSeekChatClient', 'DeepSeekReasonerClient', 'MockClient',
            'create_llm_client', 'ToolRegistry', 'registry', 'CLIApp'
        ]
        
    except ImportError:
        # If both fail, raise the original error
        raise ImportError(f"Could not import modules from new or old locations: {e}")


# Create aliases for commonly used imports
message_types = __import__(__name__ + '.message_types', fromlist=['*']) if 'Message' in globals() else None
llm_interface = __import__(__name__ + '.llm_interface', fromlist=['*']) if 'LLMClient' in globals() else None
tool_registry = __import__(__name__ + '.tool_registry', fromlist=['*']) if 'ToolRegistry' in globals() else None
cli_app = __import__(__name__ + '.cli_app', fromlist=['*']) if 'CLIApp' in globals() else None