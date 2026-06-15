import os, sys, cv2, numpy as np, time

_prj = 'D:\\PROJECTS\\AUTOAUTO'
sys.path.insert(0, _prj)
os.chdir(_prj)

# Temporarily rename dll dir so roop/__init__.py can't load wrong cuDNN
_dll = os.path.join(_prj, 'dll')
_dll_bak = os.path.join(_prj, 'dll_BACKUP')
_renamed = False
if os.path.exists(_dll):
    os.rename(_dll, _dll_bak)
    _renamed = True
    print('dll dir temporarily renamed')

try:
    import roop.globals
    from roop.ProcessMgr import ProcessMgr
    from roop.ProcessOptions import ProcessOptions
    from roop.capturer import get_video_frame
    from roop.analyser import get_face_single
    from roop.types import FaceSet
    
    src_path = r'D:\PROJECTS\uee\FACES\MARETA\mar.JPG'
    tgt_path = r'D:\PROJECTS\uee\AUTO-DEEP\input\VBIDE\MORINVIDS\1_4940920829606102276.mp4'
    debug_dir = r'D:\PROJECTS\AUTOAUTO\debug_swap'
    
    print('Loading source face...')
    src_img = cv2.imread(src_path)
    if src_img is None:
        print('ERROR: cannot read source')
        sys.exit(1)
    src_face = get_face_single(src_img)
    if src_face is None:
        print('ERROR: no face in source')
        sys.exit(1)
    print('Source face OK')
    
    # Reset class-level call counter so debug writes work (call_num <= 3)
    ProcessMgr._swap_call_count = 0
    ProcessMgr.frame_count = 0
    
    roop.globals.INPUT_FACESETS = [FaceSet(faces=[src_face])]
    roop.globals.TARGET_FACES = []
    roop.globals.distance_threshold = 0.45
    roop.globals.blend_ratio = 1.0
    roop.globals.face_swap_mode = 'selected_faces'
    roop.globals.use_enhancer = True
    roop.globals.debug = True
    roop.globals.output_path = _prj
    
    options = ProcessOptions(
        {'face_swapper': {}, 'mask_xseg': {}},
        0.45, 1.0, 'selected_faces', 0, '',
        None, 1, False, False, True, 'seamless'
    )
    
    print('Creating ProcessMgr...')
    mgr = ProcessMgr()
    print('Initializing...')
    mgr.initialize(roop.globals.INPUT_FACESETS, [], options)
    mgr._debug_first_frames = 3
    print('Init done!')
    
    for fnum in [1, 2, 3]:
        frame = get_video_frame(tgt_path, fnum)
        if frame is None:
            print('Frame {}: NONE'.format(fnum))
            continue
        h, w = frame.shape[:2]
        print('Frame {}: {}x{}'.format(fnum, w, h))
        result = mgr.process_frame(frame, file_path=tgt_path)
        if result is not None:
            outpath = os.path.join(debug_dir, 'z_final_result_f{}.png'.format(fnum))
            cv2.imwrite(outpath, result)
            print('  -> OK')
        else:
            print('  -> None')
    
    mgr.Release()
    print('DONE')
    
    # Analyze masks
    print('\nMask analysis:')
    for fnum in [1, 2, 3]:
        fp = os.path.join(debug_dir, '05_mask_final_f{}.png'.format(fnum))
        if os.path.exists(fp):
            mask = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                nz = np.count_nonzero(mask > 5)
                nz128 = np.count_nonzero(mask > 128)
                print('Frame {}: mean={:.1f}/255 ({:.4f}) >5:{} ({:.1f}%) >128:{} ({:.1f}%)'.format(
                    fnum, mask.mean(), mask.mean()/255,
                    nz, (nz/mask.size)*100,
                    nz128, (nz128/mask.size)*100))
        else:
            print('Frame {}: no mask file'.format(fnum))
    
finally:
    if _renamed and os.path.exists(_dll_bak):
        os.rename(_dll_bak, _dll)
        print('dll dir restored')
