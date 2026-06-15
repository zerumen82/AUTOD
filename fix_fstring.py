"""Fix ALL broken f-strings in ProcessMgr.py"""
import re
fpath = r'D:/PROJECTS/AUTOAUTO/roop/ProcessMgr.py'
with open(fpath, 'rb') as f:
    data = f.read()

# All broken patterns found: {video basenam} should be {video_basename}
# The pattern is the open brace followed by 'video basenam'
# We need to find all f-strings with this broken identifier
fixed_count = 0

# Iterate until no more broken patterns
while True:
    # Find broken f-string: {video basenam} -> should be {video_basename}
    broken_pattern = b"{video basenam"
    correct_pattern = b"{video_basename"
    
    # Find ALL occurrences
    positions = []
    pos = 0
    while True:
        idx = data.find(broken_pattern, pos)
        if idx == -1:
            break
        positions.append(idx)
        pos = idx + 1
    
    if not positions:
        break
    
    # Replace each occurrence
    for idx in reversed(positions):  # reversed to maintain indices
        data = data[:idx] + correct_pattern + data[idx + len(broken_pattern):]
        fixed_count += 1

print(f'Fixed {fixed_count} broken f-string patterns')

# Also check for similar patterns
other_broken = [
    (b"{video basenam}", b"{video_basename}"),
    (b"{video basenam ", b"{video_basename} "),
    (b"video basenam'", b"video_basename'"),
]

for broken, correct in other_broken:
    if broken in data:
        count = data.count(broken)
        data = data.replace(broken, correct)
        print(f'Also fixed {count} of pattern: {broken!r}')

# Write the fixed file
with open(fpath, 'wb') as f:
    f.write(data)

print('File saved.')


