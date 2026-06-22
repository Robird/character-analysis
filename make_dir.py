from character_reader import iter_characters
# from api import LLMClient
import sys
import os
import json
from itertools import islice

if __name__ == "__main__":
    character_iter = iter_characters()
    if len(sys.argv) > 1:
        max_count = int(sys.argv[1])
        character_iter = islice(character_iter, max_count)

    out_dir_path = "output"
    os.makedirs(out_dir_path)
    # client = LLMClient()
    for character in character_iter:
        dst_path = os.path.join(out_dir_path,*character.classification,character.name)
        gist_path = os.path.join(dst_path, "gist.json")
        if os.path.isfile(gist_path):
            continue

        gist_obj = {
            "name" : character.name,
            "classification" : character.classification,
            "gist" : character.gist
        }
        os.makedirs(dst_path, mode=0o777, exist_ok=True)
        with open(gist_path, mode="wt") as fp:
            json.dump(gist_obj, fp, ensure_ascii=False)