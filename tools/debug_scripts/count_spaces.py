with open('roop/img_editor/img_editor_manager.py', 'r') as f:
    lines = f.readlines()
line235 = lines[234]
line256 = lines[255]
print('Line 235 leading spaces:', len(line235) - len(line235.lstrip(' ')))
print('Line 256 leading spaces:', len(line256) - len(line256.lstrip(' ')))
