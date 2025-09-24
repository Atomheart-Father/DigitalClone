"""
Memory Management Module for DigitalClone AI Assistant.

This module implements a four-layer memory system:
- Working Buffer: Recent conversation turns and task state
- Rolling Summary: Recursive summarization of past conversations
- Semantic Memory (RAG): Vectorized knowledge from documents/web
- User Profile: Long-term user preferences and characteristics

Based on DeepSeek API characteristics and multi-turn conversation management.
"""

from .working_buffer import WorkingBuffer
from .rolling_summary import RollingSummary
from .profile_store import ProfileStore
from .rag_store import RAGStore

__all__ = [
    'WorkingBuffer',
    'RollingSummary',
    'ProfileStore',
    'RAGStore'
]
