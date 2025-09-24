"""
RAG Upsert Tool for the DigitalClone AI Assistant.

This tool inserts documents into the RAG vector database.
"""

import logging
from typing import Dict, Any, List
import json
import hashlib

logger = logging.getLogger(__name__)

# Mock RAG storage - in production, this would be ChromaDB
_rag_storage = {
    "chatlog": [],
    "web": [],
    "file": []
}

def _chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """Simple text chunking."""
    words = text.split()
    chunks = []
    current_chunk = []

    for word in words:
        current_chunk.append(word)
        if len(' '.join(current_chunk)) >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks if chunks else [text]

def run(doc_id: str, text: str, source: str = "chatlog", meta: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Execute RAG upsert tool.

    Args:
        doc_id: Document identifier
        text: Text content to store
        source: Source type (chatlog, web, file)
        meta: Additional metadata

    Returns:
        Upsert result with chunk count and status
    """
    try:
        if meta is None:
            meta = {}

        # Chunk the text
        chunks = _chunk_text(text)

        # Store chunks with metadata
        stored_chunks = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            chunk_data = {
                "id": chunk_id,
                "doc_id": doc_id,
                "text": chunk,
                "source": source,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "meta": meta,
                "embedding": f"mock_embedding_{hashlib.md5(chunk.encode()).hexdigest()[:8]}"  # Mock embedding
            }
            stored_chunks.append(chunk_data)

        # Store in mock database
        if source not in _rag_storage:
            _rag_storage[source] = []

        _rag_storage[source].extend(stored_chunks)

        logger.info(f"Upserted {len(chunks)} chunks for doc {doc_id} from source {source}")

        return {
            "ok": True,
            "chunks": len(chunks),
            "doc_id": doc_id,
            "source": source,
            "message": f"Successfully upserted {len(chunks)} chunks"
        }

    except Exception as e:
        logger.error(f"RAG upsert failed: {e}")
        return {
            "ok": False,
            "error": str(e),
            "chunks": 0,
            "doc_id": doc_id
        }

TOOL_META = {
    "name": "rag_upsert",
    "description": "将文档插入RAG向量数据库，支持文本分块和元数据存储",
    "parameters": {
        "type": "object",
        "properties": {
            "doc_id": {
                "type": "string",
                "description": "文档唯一标识符"
            },
            "text": {
                "type": "string",
                "description": "要存储的文本内容"
            },
            "source": {
                "type": "string",
                "enum": ["chatlog", "web", "file"],
                "description": "内容来源类型",
                "default": "chatlog"
            },
            "meta": {
                "type": "object",
                "description": "附加元数据",
                "default": {}
            }
        },
        "required": ["doc_id", "text"],
        "additionalProperties": False
    }
}
