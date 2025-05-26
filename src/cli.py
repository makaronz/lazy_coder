#!/usr/bin/env python3
"""
CLI wrapper for LazyCoder
"""

import sys
import os

# Add src directory to Python path
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Set PYTHONPATH environment variable
os.environ['PYTHONPATH'] = src_dir + ':' + os.environ.get('PYTHONPATH', '')

# Import and run main
try:
    from main import main
except ImportError:
    # If direct import fails, try with module path
    import subprocess
    import sys
    result = subprocess.run([
        sys.executable, '-m', 'src.main'
    ] + sys.argv[1:], cwd=os.path.dirname(src_dir))
    sys.exit(result.returncode)

def cli_main():
    main()

if __name__ == "__main__":
    cli_main()