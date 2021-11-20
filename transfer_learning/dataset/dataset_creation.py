from pathlib import Path

from jmd_imagescraper.core import *

if __name__ == "__main__":
    lasagne_musaka_burek_dir = Path().cwd() / "lasagne_musaka_tart"

    for label in ["lasagne", "musaka", "tart"]:
        duckduckgo_search(lasagne_musaka_burek_dir, label, label, max_results=1000)

    dumbell_kettlebell_barebell_dir = Path().cwd() / "dumbell_kettlebell_barebell"

    for label in ["dumbell", "kettlebell", "barebell"]:
        duckduckgo_search(
            dumbell_kettlebell_barebell_dir, label, label, max_results=1000
        )
