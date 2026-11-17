"""Fetch the finished results/ (no training needed for the notebooks). Maintainers: --pack OUT builds the release tarball."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from m5knobs.download import main  # noqa: E402

if __name__ == "__main__":
    main(sys.argv[1:])
