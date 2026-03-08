import ctypes, os
capi = r'C:\Users\INdaHouse\AppData\Local\Programs\Python\Python310\Lib\site-packages\onnxruntime\capi'
for name in os.listdir(capi):
    if name.lower().startswith('cudnn') and name.lower().endswith('.dll'):
        path = os.path.join(capi, name)
        try:
            ctypes.WinDLL(path)
            print('Loaded:', name)
        except Exception as e:
            print('Failed to load', name, '->', e)
