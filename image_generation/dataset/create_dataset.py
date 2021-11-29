import sys
from pathlib import Path

from jmd_imagescraper.core import *


if __name__ == "__main__":
    _script, class_name, images_count = sys.argv
    path = Path.cwd() / "data"

    duckduckgo_search(path, class_name, class_name, max_results=int(images_count))
