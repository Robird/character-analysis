#!/usr/bin/env python3
"""规范脚本：将单行多人物（/ 在括号外）拆分为每行一人"""
import re
import sys
from pathlib import Path

def split_line(line: str) -> list[str]:
    """返回拆分后的多行（如果不需要拆分则返回原行）
    
    规则：bullet 行含 /，但名字段（- 到 — 之间）有中文括号或英文括号时，
    / 是同一角色的多语种别名，不拆。否则是多角色合并，拆分。
    """
    stripped = line.lstrip()
    if not stripped.startswith('- '):
        return [line]

    dash_idx = line.index('—') if '—' in line else len(line)
    body = line[:dash_idx]
    desc = line[dash_idx:] if dash_idx < len(line) else ''

    # 名字字段 = -  之后到 — 之前
    body_content = body[2:].strip()

    # 有 / 才需要处理
    if ' / ' not in body_content and '/' not in body_content:
        return [line]

    # 关键判断：名字段里有任何括号 → 同一角色的翻译别名，不拆
    if any(ch in body_content for ch in '（(）)'):
        return [line]

    # 无括号 → 拆！
    indent = line[:len(line) - len(stripped)]
    prefix = indent + '- '

    parts = body_content.split(' / ')
    if len(parts) == 1:
        parts = body_content.split('/')
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) <= 1:
        return [line]

    result = []
    for part in parts:
        result.append(f"{prefix}{part}{desc}")
    return result

def process_file(filepath: Path) -> int:
    """处理单个文件，返回修改的行数"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    new_lines = []
    changed = 0
    for line in lines:
        split = split_line(line.rstrip('\n'))
        if len(split) > 1:
            changed += 1
        for s in split:
            new_lines.append(s + '\n')
    
    if changed > 0:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)
    
    return changed

def main():
    base = Path('/repos/character-analysis/characters')
    md_files = sorted(base.rglob('*.md'))
    
    total_changed = 0
    total_lines = 0
    for fp in md_files:
        c = process_file(fp)
        if c > 0:
            print(f"  ✓ {fp.relative_to(base)} — {c} 行拆分")
            total_changed += c
            total_lines += 1
    
    print(f"\n总计: {total_lines} 个文件, {total_changed} 行已拆分")

if __name__ == '__main__':
    main()
