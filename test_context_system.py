#!/usr/bin/env python3
"""
Test script for the new context management system.

Tests the four-layer memory system and context assembler integration.
"""

import sys
import os

# Add project paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_memory_layers():
    """Test individual memory layer functionality."""
    from memory.working_buffer import WorkingBuffer
    from memory.rolling_summary import RollingSummary
    from memory.profile_store import ProfileStore
    from memory.rag_store import RAGStore

    print("🧪 Testing Memory Layers...")

    # Test Working Buffer
    buffer = WorkingBuffer(max_tokens=1000, max_turns=5)
    buffer.append_turn("user", "Hello, I need help with a task")
    buffer.append_turn("assistant", "I'd be happy to help. What do you need?")

    assert len(buffer.turns) == 2
    print("✅ Working Buffer: Basic functionality works")

    # Test Rolling Summary
    summary = RollingSummary(max_tokens=500)
    summary.update_summary("Previous conversation content...")
    assert len(summary.get_summary()) > 0
    print("✅ Rolling Summary: Basic functionality works")

    # Test Profile Store
    profile = ProfileStore(storage_path="test_profile.json")
    profile.upsert_profile_fact("writing_style", "concise", 0.8)
    assert profile.get_profile_fact("writing_style") == "concise"
    profile.clear()
    os.remove("test_profile.json")
    print("✅ Profile Store: Basic functionality works")

    # Test RAG Store
    rag = RAGStore(storage_path="test_rag.json", max_chunks=10)
    rag.add_document("This is a test document with some content.", "test_source")
    results = rag.search("test document")
    assert len(results) > 0
    rag.clear()
    os.remove("test_rag.json")
    print("✅ RAG Store: Basic functionality works")

def test_context_assembler():
    """Test context assembler integration."""
    from context.assembler import ContextAssembler
    from memory.working_buffer import WorkingBuffer

    print("🧪 Testing Context Assembler...")

    # Create components
    buffer = WorkingBuffer(max_tokens=1000)
    buffer.append_turn("user", "Test query")

    assembler = ContextAssembler(working_buffer=buffer)

    # Test assembly
    result = assembler.assemble("Test query", budget_tokens=2000)

    assert 'messages' in result
    assert 'metadata' in result
    assert len(result['messages']) > 0

    metadata = result['metadata']
    assert 'total_chars' in metadata
    assert 'estimated_tokens' in metadata
    assert 'section_breakdown' in metadata

    print("✅ Context Assembler: Basic functionality works")
    print(f"   Assembled {len(result['messages'])} messages, ~{metadata['estimated_tokens']} tokens")

def test_micro_decider():
    """Test micro-decision functionality."""
    from reasoner.micro_decide import MicroDecider

    print("🧪 Testing Micro Decider...")

    decider = MicroDecider(max_tokens=100, timeout_seconds=10)

    # Mock test (since we don't have API keys in test)
    stats = decider.get_stats()
    assert stats['max_tokens'] == 100
    assert stats['timeout_seconds'] == 10

    print("✅ Micro Decider: Basic structure works")

def test_integration():
    """Test full integration of context management system."""
    from context.assembler import ContextAssembler
    from memory.working_buffer import WorkingBuffer
    from memory.rolling_summary import RollingSummary

    print("🧪 Testing Full Integration...")

    # Create integrated system
    buffer = WorkingBuffer(max_tokens=2000, max_turns=10)
    summary = RollingSummary(max_tokens=500)

    assembler = ContextAssembler(
        working_buffer=buffer,
        rolling_summary=summary
    )

    # Simulate conversation
    buffer.append_turn("user", "I need help with data analysis")
    buffer.append_turn("assistant", "I'd be happy to help. What kind of data do you have?")

    # Add some "dropped" content to test summary
    assembler.update_memories([], "Previous analysis conversation...")

    # Test assembly
    result = assembler.assemble("How to analyze this data?", budget_tokens=3000)

    assert len(result['messages']) >= 2  # At least system + conversation
    assert result['metadata']['estimated_tokens'] < 3000

    print("✅ Full Integration: Context management system works")
    print(f"   Final assembly: {result['metadata']['message_count']} messages, {result['metadata']['estimated_tokens']} tokens")

if __name__ == "__main__":
    print("🚀 Testing Context Management System")
    print("=" * 50)

    try:
        test_memory_layers()
        print()

        test_context_assembler()
        print()

        test_micro_decider()
        print()

        test_integration()
        print()

        print("🎉 All tests passed! Context management system is working correctly.")
        print("📋 Ready for production use with proper API keys.")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
