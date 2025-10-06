#!/usr/bin/env python3
"""
New startup script for the Digital Clone AI Assistant.

This script uses the new directory structure and provides backward compatibility.
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

# Try to import from new structure first, then fallback to old structure
try:
    from apps.cli import main
    print("✓ Using new directory structure")
except ImportError as e:
    print(f"⚠️  New structure not available: {e}")
    try:
        # Fallback to old structure
        from backend.cli_app import main
        print("✓ Using backward compatibility mode")
    except ImportError as e2:
        print(f"❌ Both new and old structures failed: {e2}")
        sys.exit(1)

if __name__ == "__main__":
    main()
