"""
User Profile Store - Long-term user preferences and characteristics.

This implements persistent storage for user preferences, writing styles,
communication patterns, and other long-term characteristics.
"""

import json
import time
from typing import Dict, Any, Optional, List
from pathlib import Path

# Conditional imports for configuration
try:
    from backend.config import Config
except ImportError:
    Config = None


class ProfileFact:
    """A single user profile fact with metadata."""

    def __init__(self, key: str, value: Any, confidence: float = 0.5,
                 source: str = "conversation", ttl_days: Optional[int] = None):
        """
        Initialize a profile fact.

        Args:
            key: Fact identifier (e.g., 'writing_style', 'preferred_format')
            value: Fact value
            confidence: Confidence score (0.0 to 1.0)
            source: Source of the fact (conversation, explicit_setting, etc.)
            ttl_days: Time-to-live in days (None for permanent)
        """
        self.key = key
        self.value = value
        self.confidence = confidence
        self.source = source
        self.created_at = time.time()
        self.updated_at = time.time()
        self.ttl_days = ttl_days
        self.access_count = 0
        self.last_accessed = time.time()

    def is_expired(self) -> bool:
        """Check if the fact has expired."""
        if self.ttl_days is None:
            return False

        expiry_time = self.created_at + (self.ttl_days * 24 * 60 * 60)
        return time.time() > expiry_time

    def touch(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.last_accessed = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'key': self.key,
            'value': self.value,
            'confidence': self.confidence,
            'source': self.source,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'ttl_days': self.ttl_days,
            'access_count': self.access_count,
            'last_accessed': self.last_accessed
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProfileFact':
        """Create from dictionary."""
        fact = cls(
            key=data['key'],
            value=data['value'],
            confidence=data.get('confidence', 0.5),
            source=data.get('source', 'conversation'),
            ttl_days=data.get('ttl_days')
        )
        fact.created_at = data.get('created_at', time.time())
        fact.updated_at = data.get('updated_at', time.time())
        fact.access_count = data.get('access_count', 0)
        fact.last_accessed = data.get('last_accessed', time.time())
        return fact


class ProfileStore:
    """
    Persistent storage for user profile information.

    Maintains long-term user characteristics, preferences, and patterns
    that influence conversation and task execution.
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize profile store.

        Args:
            storage_path: Path to store profile data (uses default if None)
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
            self.storage_path = base_dir / "profile.json"

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.facts: Dict[str, ProfileFact] = {}
        self.load()

    def upsert_profile_fact(self, key: str, value: Any,
                           confidence: float = 0.5, source: str = "conversation",
                           ttl_days: Optional[int] = None) -> None:
        """
        Insert or update a user profile fact.

        Args:
            key: Fact identifier
            value: Fact value
            confidence: Confidence score (0.0 to 1.0)
            source: Source of the information
            ttl_days: Time-to-live in days (None for permanent)
        """
        if key in self.facts:
            # Update existing fact
            fact = self.facts[key]
            fact.value = value
            fact.confidence = max(fact.confidence, confidence)  # Keep higher confidence
            fact.source = source
            fact.updated_at = time.time()
            fact.ttl_days = ttl_days
        else:
            # Create new fact
            fact = ProfileFact(key, value, confidence, source, ttl_days)
            self.facts[key] = fact

        # Auto-save
        self.save()

    def get_profile_fact(self, key: str) -> Optional[Any]:
        """
        Get a user profile fact.

        Args:
            key: Fact identifier

        Returns:
            Fact value or None if not found/expired
        """
        if key not in self.facts:
            return None

        fact = self.facts[key]

        # Check if expired
        if fact.is_expired():
            del self.facts[key]
            self.save()
            return None

        # Update access stats
        fact.touch()
        return fact.value

    def get_all_facts(self, include_expired: bool = False) -> Dict[str, Any]:
        """
        Get all profile facts.

        Args:
            include_expired: Whether to include expired facts

        Returns:
            Dictionary of fact key-value pairs
        """
        result = {}
        to_remove = []

        for key, fact in self.facts.items():
            if fact.is_expired():
                if not include_expired:
                    to_remove.append(key)
                else:
                    result[key] = fact.value
            else:
                result[key] = fact.value

        # Clean up expired facts
        for key in to_remove:
            del self.facts[key]

        if to_remove:
            self.save()

        return result

    def remove_fact(self, key: str) -> bool:
        """
        Remove a profile fact.

        Args:
            key: Fact identifier

        Returns:
            True if fact was removed, False if not found
        """
        if key in self.facts:
            del self.facts[key]
            self.save()
            return True
        return False

    def get_fact_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a profile fact.

        Args:
            key: Fact identifier

        Returns:
            Fact metadata dictionary or None if not found
        """
        if key not in self.facts:
            return None

        fact = self.facts[key]
        if fact.is_expired():
            return None

        return {
            'confidence': fact.confidence,
            'source': fact.source,
            'created_at': fact.created_at,
            'updated_at': fact.updated_at,
            'ttl_days': fact.ttl_days,
            'access_count': fact.access_count,
            'last_accessed': fact.last_accessed,
            'is_expired': False
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get profile store statistics."""
        total_facts = len(self.facts)
        expired_count = sum(1 for fact in self.facts.values() if fact.is_expired())
        active_count = total_facts - expired_count

        # Category breakdown
        categories = {}
        for fact in self.facts.values():
            if not fact.is_expired():
                category = fact.key.split('_')[0]  # Simple categorization
                categories[category] = categories.get(category, 0) + 1

        return {
            'total_facts': total_facts,
            'active_facts': active_count,
            'expired_facts': expired_count,
            'categories': categories,
            'storage_path': str(self.storage_path)
        }

    def clear_expired(self) -> int:
        """
        Remove all expired facts.

        Returns:
            Number of expired facts removed
        """
        to_remove = [key for key, fact in self.facts.items() if fact.is_expired()]
        for key in to_remove:
            del self.facts[key]

        if to_remove:
            self.save()

        return len(to_remove)

    def clear(self) -> None:
        """Clear all profile facts."""
        self.facts.clear()
        self.save()

    def save(self) -> None:
        """Save profile facts to disk."""
        try:
            data = {key: fact.to_dict() for key, fact in self.facts.items()}
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Failed to save profile store: {e}")

    def load(self) -> None:
        """Load profile facts from disk."""
        try:
            if self.storage_path.exists():
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                self.facts = {}
                for key, fact_data in data.items():
                    try:
                        fact = ProfileFact.from_dict(fact_data)
                        self.facts[key] = fact
                    except Exception as e:
                        print(f"Failed to load profile fact {key}: {e}")
        except Exception as e:
            print(f"Failed to load profile store: {e}")
            self.facts = {}
