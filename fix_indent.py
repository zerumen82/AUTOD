import sys

file_path = 'roop/ProcessMgr.py'
with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

cleaned = [line.rstrip() + '\n' for line in lines]
while cleaned and not cleaned[-1].strip():
    cleaned.pop()

with open(file_path, 'w', encoding='utf-8') as f:
    f.writelines(cleaned)
