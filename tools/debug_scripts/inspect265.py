with open('roop/img_editor/img_editor_manager.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for i in range(263, 270):
    line = lines[i]
    spaces = len(line) - len(line.lstrip(' '))
    print(f'{i+1:4d} ({spaces:2d}): {repr(line[:120])}')
