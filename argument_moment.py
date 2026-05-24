from character_reader import iter_characters
from api import LLMClient
import sys
from itertools import islice

if __name__ == "__main__":
    max_lines = 1  # 限制输出，防止工具截断
    if len(sys.argv) > 1:
        max_lines = int(sys.argv[1])

    client = LLMClient()