#!/usr/bin/env python3
"""
Test runner script for the DigitalClone AI Assistant.

This script sets up the correct Python path for running tests.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Add the backend directory to Python path
backend_dir = project_root / "backend"
sys.path.insert(0, str(backend_dir))

if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main(sys.argv[1:]))
