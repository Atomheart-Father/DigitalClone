"""
RAG Store - Semantic memory using vectorized document storage.

This implements the semantic memory layer that stores and retrieves
document content using vector similarity search.
"""

import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

# Conditional imports
try:
    from backend.config import Config
except ImportError:
    Config = None


@dataclass
class DocumentChunk:
    """A document chunk with metadata."""

    id: str
    content: str
    source: str
    chunk_index: int
    total_chunks: int
    metadata: Dict[str, Any]
    fingerprint: str  # For deduplication
    created_at: float

    @property
    def full_source(self) -> str:
        """Get the full source identifier."""
        return f"{self.source}#chunk_{self.chunk_index}"


class RAGStore:
    """
    Vectorized document storage for semantic memory.

    Stores document chunks and provides similarity-based retrieval
    for contextual information augmentation.
    """

    def __init__(self, storage_path: Optional[str] = None, max_chunks: int = 1000):
        """
        Initialize RAG store.

        Args:
            storage_path: Path to store document data
            max_chunks: Maximum number of chunks to store
        """
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            # Default to data directory
            try:
                if Config and hasattr(Config, 'LOG_DIR'):
                    base_dir = Path(Config.LOG_DIR).parent
                else:
                    base_dir = Path("data")
            except:
                base_dir = Path("data")
            self.storage_path = base_dir / "rag_store.json"

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.max_chunks = max_chunks
        self.chunks: List[DocumentChunk] = []
        self.load()

    def add_document(self, content: str, source: str,
                    metadata: Optional[Dict[str, Any]] = None,
                    chunk_size: int = 1000, overlap: int = 200) -> int:
        """
        Add a document to the RAG store with chunking.

        Args:
            content: Document content
            source: Source identifier (e.g., file path, URL)
            metadata: Additional metadata
            chunk_size: Size of each chunk in characters
            overlap: Overlap between chunks

        Returns:
            Number of chunks added
        """
        if not content.strip():
            return 0

        chunks = self._chunk_text(content, chunk_size, overlap)
        added_count = 0

        for i, chunk_content in enumerate(chunks):
            # Create fingerprint for deduplication
            fingerprint = self._create_fingerprint(chunk_content)

            # Check for duplicates
            if not self._is_duplicate(fingerprint):
                chunk = DocumentChunk(
                    id=f"{source}_{i}",
                    content=chunk_content,
                    source=source,
                    chunk_index=i,
                    total_chunks=len(chunks),
                    metadata=metadata or {},
                    fingerprint=fingerprint,
                    created_at=time.time()
                )

                self.chunks.append(chunk)
                added_count += 1

        # Enforce limits
        self._enforce_limits()

        # Auto-save
        self.save()

        return added_count

    def search(self, query: str, k: int = 4,
              filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant document chunks.

        Args:
            query: Search query
            k: Number of results to return
            filters: Optional filters for source/metadata

        Returns:
            List of relevant chunks with scores
        """
        if not self.chunks:
            return []

        # Simple keyword-based search (can be upgraded to vector search)
        scored_chunks = []

        query_lower = query.lower()
        query_terms = set(query_lower.split())

        for chunk in self.chunks:
            # Apply filters
            if filters:
                if not self._matches_filters(chunk, filters):
                    continue

            # Calculate relevance score
            score = self._calculate_relevance(chunk.content.lower(), query_terms)
            if score > 0:
                scored_chunks.append({
                    'chunk': chunk,
                    'score': score,
                    'content': chunk.content,
                    'source': chunk.source,
                    'metadata': chunk.metadata
                })

        # Sort by score and return top k
        scored_chunks.sort(key=lambda x: x['score'], reverse=True)
        return scored_chunks[:k]

    def remove_document(self, source: str) -> int:
        """
        Remove all chunks from a specific source.

        Args:
            source: Source identifier

        Returns:
            Number of chunks removed
        """
        original_count = len(self.chunks)
        self.chunks = [chunk for chunk in self.chunks if chunk.source != source]
        removed_count = original_count - len(self.chunks)

        if removed_count > 0:
            self.save()

        return removed_count

    def get_stats(self) -> Dict[str, Any]:
        """Get RAG store statistics."""
        total_chars = sum(len(chunk.content) for chunk in self.chunks)
        sources = set(chunk.source for chunk in self.chunks)

        # Source breakdown
        source_counts = {}
        for chunk in self.chunks:
            source_counts[chunk.source] = source_counts.get(chunk.source, 0) + 1

        return {
            'total_chunks': len(self.chunks),
            'total_chars': total_chars,
            'unique_sources': len(sources),
            'source_breakdown': source_counts,
            'max_chunks': self.max_chunks,
            'utilization_percent': (len(self.chunks) / self.max_chunks) * 100,
            'storage_path': str(self.storage_path)
        }

    def clear(self) -> None:
        """Clear all stored chunks."""
        self.chunks.clear()
        self.save()

    def save(self) -> None:
        """Save chunks to disk."""
        try:
            data = [self._chunk_to_dict(chunk) for chunk in self.chunks]
            import json
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save RAG store: {e}")

    def load(self) -> None:
        """Load chunks from disk."""
        try:
            if self.storage_path.exists():
                import json
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                self.chunks = []
                for item in data:
                    try:
                        chunk = self._dict_to_chunk(item)
                        self.chunks.append(chunk)
                    except Exception as e:
                        print(f"Failed to load chunk: {e}")
        except Exception as e:
            print(f"Failed to load RAG store: {e}")
            self.chunks = []

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """
        Split text into overlapping chunks.

        Args:
            text: Text to chunk
            chunk_size: Size of each chunk
            overlap: Overlap between chunks

        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + chunk_size

            # Find a good breaking point (sentence end)
            if end < len(text):
                # Look for sentence endings in the last 100 chars
                search_start = max(start + chunk_size - 100, start)
                sentence_end = text.rfind('.', search_start, end)
                if sentence_end == -1:
                    sentence_end = text.rfind('!', search_start, end)
                if sentence_end == -1:
                    sentence_end = text.rfind('?', search_start, end)
                if sentence_end == -1:
                    sentence_end = text.rfind('\n', search_start, end)

                if sentence_end > search_start:
                    end = sentence_end + 1

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start position with overlap
            start = end - overlap
            if start >= len(text):
                break

        return chunks

    def _create_fingerprint(self, content: str) -> str:
        """
        Create a fingerprint for deduplication.

        Args:
            content: Content to fingerprint

        Returns:
            SHA256 fingerprint
        """
        # Normalize content for deduplication
        normalized = content.lower().strip()
        return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

    def _is_duplicate(self, fingerprint: str) -> bool:
        """
        Check if a fingerprint already exists.

        Args:
            fingerprint: Content fingerprint

        Returns:
            True if duplicate exists
        """
        return any(chunk.fingerprint == fingerprint for chunk in self.chunks)

    def _calculate_relevance(self, content: str, query_terms: set) -> float:
        """
        Calculate relevance score between content and query.

        Args:
            content: Document content
            query_terms: Set of query terms

        Returns:
            Relevance score (0.0 to 1.0)
        """
        if not query_terms:
            return 0.0

        # Simple term frequency scoring
        content_terms = set(content.split())
        matches = query_terms.intersection(content_terms)

        if not matches:
            return 0.0

        # Calculate score based on match ratio and term frequency
        match_ratio = len(matches) / len(query_terms)
        term_frequency = sum(content.count(term) for term in matches)

        # Boost score for multiple occurrences
        frequency_boost = min(1.0, term_frequency / len(query_terms))

        return match_ratio * (0.7 + 0.3 * frequency_boost)

    def _matches_filters(self, chunk: DocumentChunk, filters: Dict[str, Any]) -> bool:
        """
        Check if a chunk matches the given filters.

        Args:
            chunk: Document chunk
            filters: Filter criteria

        Returns:
            True if chunk matches all filters
        """
        for key, value in filters.items():
            if key == 'source':
                if chunk.source != value:
                    return False
            elif key in chunk.metadata:
                if chunk.metadata[key] != value:
                    return False
            else:
                return False
        return True

    def _enforce_limits(self) -> None:
        """Enforce chunk limits by removing oldest chunks."""
        if len(self.chunks) > self.max_chunks:
            # Keep most recent chunks
            excess = len(self.chunks) - self.max_chunks
            self.chunks = self.chunks[excess:]
            self.save()

    def _chunk_to_dict(self, chunk: DocumentChunk) -> Dict[str, Any]:
        """Convert chunk to dictionary for serialization."""
        return {
            'id': chunk.id,
            'content': chunk.content,
            'source': chunk.source,
            'chunk_index': chunk.chunk_index,
            'total_chunks': chunk.total_chunks,
            'metadata': chunk.metadata,
            'fingerprint': chunk.fingerprint,
            'created_at': chunk.created_at
        }

    def _dict_to_chunk(self, data: Dict[str, Any]) -> DocumentChunk:
        """Convert dictionary to chunk."""
        return DocumentChunk(
            id=data['id'],
            content=data['content'],
            source=data['source'],
            chunk_index=data.get('chunk_index', 0),
            total_chunks=data.get('total_chunks', 1),
            metadata=data.get('metadata', {}),
            fingerprint=data.get('fingerprint', ''),
            created_at=data.get('created_at', time.time())
        )
