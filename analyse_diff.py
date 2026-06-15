import cv2, numpy as np
import sys

d = 'debug_swap'
fnum = int(sys.argv[1])

res = cv2.imread(f'{d}/04_final_result_f{fnum}.png')
orig = cv2.imread(f'{d}/07_original_frame_f{fnum}.png')
warp = cv2.imread(f'{d}/06_warped_face_f{fnum}.png')
mask = cv2.imread(f'{d}/05_mask_final_f{fnum}.png', cv2.IMREAD_GRAYSCALE)
blended = cv2.imread(f'{d}/08_warped_final_mask_blended_f{fnum}.png')

diff = cv2.absdiff(res, orig)
print(f'  diff medio: {diff.mean():.2f}, diff max: {diff.max():.0f}')

if mask is not None and mask.sum() > 0:
    mask_bool = mask > 0
    mask_half = mask > 200  # core face
    mask_transition = mask_bool & ~mask_half
    diff_in_mask = diff[mask_bool].mean()
    diff_core = diff[mask_half].mean()
    diff_trans = diff[mask_transition].mean() if mask_transition.sum() > 0 else 0
    diff_out_mask = diff[~mask_bool].mean()
    m_mean = mask.mean()
    print(f'  diff en mask core: {diff_core:.2f}')
    print(f'  diff en transition: {diff_trans:.2f}')
    print(f'  diff fuera del mask: {diff_out_mask:.2f}')

# edge check
res_gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
res_edge = cv2.Canny(res_gray, 30, 100)
orig_edge = cv2.Canny(orig_gray, 30, 100)
new_edges = res_edge.astype(bool) & ~orig_edge.astype(bool)
print(f'  new edges vs orig: {new_edges.sum()}')

if mask is not None:
    gx = cv2.Sobel(mask.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(mask.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx**2 + gy**2)
    bound_mask = (grad > 2).astype(np.uint8)
    bound_mask = cv2.dilate(bound_mask, None, iterations=3)
    edges_at_bound = new_edges & bound_mask.astype(bool)
    print(f'  new edges at mask boundary: {edges_at_bound.sum()}')

# Check blended vs orig
if blended is not None:
    bd = cv2.absdiff(blended, orig)
    bd_mean = bd.mean()
    bd_max = bd.max()
    print(f'  blended vs orig diff: {bd_mean:.2f} max: {bd_max:.0f}')

# Save amplified diff
diff4 = (diff * 4).clip(0, 255).astype(np.uint8)
cv2.imwrite(f'{d}/debug_diff_amp4_f{fnum}.png', diff4)
print(f'  saved debug_diff_amp4_f{fnum}.png')
