"""
RAG Search Tool for the DigitalClone AI Assistant.

This tool searches the RAG vector database for relevant documents.
"""

import logging
from typing import Dict, Any, List
import re

logger = logging.getLogger(__name__)

# Import the mock storage from upsert tool
try:
    from .tool_rag_upsert import _rag_storage
except ImportError:
    # Fallback for when imported independently
    _rag_storage = {
        "chatlog": [],
        "web": [],
        "file": []
    }

def _calculate_similarity(query: str, text: str) -> float:
    """Simple text similarity calculation (mock implementation)."""
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())

    # Jaccard similarity
    intersection = len(query_words & text_words)
    union = len(query_words | text_words)

    return intersection / union if union > 0 else 0.0

def run(query: str, k: int = 5, source: str = "any") -> Dict[str, Any]:
    """
    Execute RAG search tool.

    Args:
        query: Search query
        k: Number of results to return
        source: Source to search in (any, chatlog, web, file)

    Returns:
        Search results with scores and metadata
    """
    try:
        results = []

        # Determine which sources to search
        sources_to_search = []
        if source == "any":
            sources_to_search = ["chatlog", "web", "file"]
        else:
            sources_to_search = [source]

        # Search each source
        for src in sources_to_search:
            if src not in _rag_storage:
                continue

            for chunk in _rag_storage[src]:
                score = _calculate_similarity(query, chunk["text"])

                if score > 0:  # Only include relevant results
                    results.append({
                        "score": score,
                        "text": chunk["text"],
                        "doc_id": chunk["doc_id"],
                        "source": chunk["source"],
                        "chunk_index": chunk["chunk_index"],
                        "total_chunks": chunk["total_chunks"],
                        "meta": chunk["meta"]
                    })

        # Sort by score and take top k
        results.sort(key=lambda x: x["score"], reverse=True)
        top_results = results[:k]

        logger.info(f"RAG search for '{query}' returned {len(top_results)} results from {source}")

        return {
            "ok": True,
            "query": query,
            "results": top_results,
            "total_found": len(results),
            "returned": len(top_results),
            "sources": sources_to_search
        }

    except Exception as e:
        logger.error(f"RAG search failed: {e}")
        return {
            "ok": False,
            "error": str(e),
            "query": query,
            "results": [],
            "total_found": 0,
            "returned": 0
        }

TOOL_META = {
    "name": "rag_search",
    "description": "在RAG向量数据库中搜索相关文档片段",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索查询字符串"
            },
            "k": {
                "type": "integer",
                "description": "返回结果数量",
                "default": 5,
                "minimum": 1,
                "maximum": 20
            },
            "source": {
                "type": "string",
                "enum": ["any", "chatlog", "web", "file"],
                "description": "搜索范围",
                "default": "any"
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}
