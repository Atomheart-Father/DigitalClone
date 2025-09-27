"""
Context Management Module for DigitalClone AI Assistant.

This module implements the four-layer memory system and intelligent
context assembly for multi-turn conversations.
"""

from .assembler import ContextAssembler
from .compressor import TextCompressor, CompressionResult

__all__ = [
    'ContextAssembler',
    'TextCompressor',
    'CompressionResult'
]
