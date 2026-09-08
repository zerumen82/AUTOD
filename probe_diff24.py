import cv2, numpy as np, os, glob, json

pairs = []
for inp in glob.glob(r".autodeep_temp\**\*", recursive=True):
    if not os.path.isfile(inp):
        continue
    name = os.path.basename(inp)
    if name.endswith(".jpeg"):
        out = os.path.join("output", name[:-5] + ".jpg")
    elif name.endswith(".png"):
        out = os.path.join("output", name)
    else:
        out = os.path.join("output", name)
    if os.path.isfile(out):
        pairs.append((inp, out))

print(f"pairs: {len(pairs)}")
results = []
for inp, out in sorted(pairs, key=lambda p: os.path.basename(p[0])):
    a = cv2.imread(inp)
    b = cv2.imread(out)
    if a is None or b is None:
        print(os.path.basename(inp), "READ FAIL", a is None, b is None)
        continue
    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))
    d = np.abs(a.astype(np.float32) - b.astype(np.float32))
    results.append((os.path.basename(inp), round(float(d.mean()), 3), round(float(d.max()), 1), a.shape[1], a.shape[0]))

print(f"{'file':28s} {'mean':>7s} {'max':>7s}  dims")
for r in sorted(results, key=lambda x: x[1]):
    print(f"{r[0]:28s} {r[1]:7.3f} {r[2]:7.1f}  {r[3]}x{r[4]}")
