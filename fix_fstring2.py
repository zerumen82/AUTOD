"""Fix broken f-strings - uses raw bytes to avoid shell quoting issues"""
fpath = r'D:/PROJECTS/AUTOAUTO/roop/ProcessMgr.py'

with open(fpath, 'rb') as f:
    data = f.read()

# Build the broken pattern as bytes to avoid shell interpretation
# The pattern in the file is: {video basenam}  (with space between video and basenam)
broken = b"video basenam"
correct = b"video_basename"

count = data.count(broken)
print(f'Found {count} occurrences of broken pattern')

if count > 0:
    new_data = data.replace(broken, correct)
    with open(fpath, 'wb') as f:
        f.write(new_data)
    print('File fixed successfully')
else:
    print('Pattern not found - showing hex of nearby content:')
    idx = data.find(b'_tracking_lost_count_')
    if idx >= 0:
        print(repr(data[idx:idx+60]))