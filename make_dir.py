from character_reader import iter_characters
from character_gist import CharacterGist
# from api import LLMClient
import sys
from itertools import islice
from pathlib import Path

if __name__ == "__main__":
    character_iter = iter_characters()
    if len(sys.argv) > 1:
        max_count = int(sys.argv[1])
        character_iter = islice(character_iter, max_count)

    out_dir_path = Path("output")
    out_dir_path.mkdir(exist_ok=True)
    # client = LLMClient()
    for character in character_iter:
        dst_path = out_dir_path.joinpath(*character.classification, character.name)
        if (dst_path / "gist.json").is_file():
            continue
        CharacterGist.SaveToJson(character, dst_path)