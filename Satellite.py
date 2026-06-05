"""
Backward-compatibility entry point.

The implementation has been moved into the `aeos/` package.
Run `python main.py` for the primary entry point.
"""

from aeos import *  # noqa: F401, F403
from main import main

if __name__ == "__main__":
    main()
