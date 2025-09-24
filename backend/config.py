"""
Configuration management for the Digital Clone AI Assistant.

This module handles loading configuration from environment variables
and provides centralized access to all system settings.
"""

import os
from typing import Optional
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not available, continue with system environment
    pass


class Config:
    """Central configuration class for the application."""

    # DeepSeek API Configuration
    DEEPSEEK_API_KEY: Optional[str] = os.getenv("DEEPSEEK_API_KEY")
    DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

    # Model Configuration (support both old and new env var names)
    MODEL_CHAT: str = os.getenv("MODEL_CHAT", os.getenv("CHAT_MODEL", "deepseek-chat"))
    MODEL_REASONER: str = os.getenv("MODEL_REASONER", os.getenv("REASONING_MODEL", "deepseek-reasoner"))

    # System Configuration
    TIMEOUT_SECONDS_CHAT: int = int(os.getenv("TIMEOUT_SECONDS_CHAT", "30"))  # Chat model timeout
    TIMEOUT_SECONDS_REASONER: int = int(os.getenv("TIMEOUT_SECONDS_REASONER", "120"))  # Reasoner model timeout (longer for complex reasoning)
    MAX_TOOL_CALLS_PER_TURN: int = int(os.getenv("MAX_TOOL_CALLS_PER_TURN", "3"))
    MAX_ASK_USER_CYCLES: int = int(os.getenv("MAX_ASK_USER_CYCLES", "2"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    # Reasoner Micro-Inference Configuration
    REASONER_MICRO_CONNECT_TIMEOUT: int = int(os.getenv("REASONER_MICRO_CONNECT_TIMEOUT", "10"))  # Connect timeout for micro-inference
    REASONER_MICRO_READ_TIMEOUT: int = int(os.getenv("REASONER_MICRO_READ_TIMEOUT", "30"))  # Read timeout for micro-inference
    REASONER_MICRO_MAX_INPUT_CHARS: int = int(os.getenv("REASONER_MICRO_MAX_INPUT_CHARS", "200"))  # Max input chars for micro-inference
    REASONER_MICRO_MAX_CANDIDATES: int = int(os.getenv("REASONER_MICRO_MAX_CANDIDATES", "3"))  # Max candidates for micro-inference

    # Backward compatibility
    TIMEOUT_SECONDS: int = TIMEOUT_SECONDS_CHAT

    # Data Paths
    LOG_DIR: str = os.getenv("LOG_DIR", "data/logs")

    # Development Settings
    ENABLE_MOCK_CLIENT_IF_NO_KEY: bool = os.getenv("ENABLE_MOCK_CLIENT_IF_NO_KEY", "true").lower() == "true"

    @classmethod
    def ensure_log_directory(cls) -> Path:
        """Ensure the log directory exists and return its path."""
        log_path = Path(cls.LOG_DIR)
        log_path.mkdir(parents=True, exist_ok=True)
        return log_path

    @classmethod
    def is_api_key_available(cls) -> bool:
        """Check if DeepSeek API key is available."""
        return cls.DEEPSEEK_API_KEY is not None and len(cls.DEEPSEEK_API_KEY.strip()) > 0

    @classmethod
    def should_use_mock_client(cls) -> bool:
        """Determine if mock client should be used."""
        return cls.ENABLE_MOCK_CLIENT_IF_NO_KEY and not cls.is_api_key_available()

    @classmethod
    def get_log_file_path(cls, date_str: str) -> Path:
        """Get the log file path for a specific date."""
        return cls.ensure_log_directory() / f"{date_str}.jsonl"


# Global config instance
config = Config()
