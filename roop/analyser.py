import insightface  
import roop.globals  
import os
import torch

FACE_ANALYSER = None  

def get_face_analyser():  
    global FACE_ANALYSER  
    if FACE_ANALYSER is None:  
        # Asegurar que los modelos se carguen desde la ruta correcta
        models_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'buffalo_l')
        
        # Usar torch.cuda para verificar GPU (más confiable que onnxruntime providers)
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print(f"[INFO] GPU detectada - Usando CUDA para FaceAnalysis")
        else:
            providers = ['CPUExecutionProvider']
            print(f"[INFO] Sin GPU - Usando CPU para FaceAnalysis")
        
        print(f"[INFO] Inicializando FaceAnalysis con providers: {providers}")
        
        try:
            # SIN provider_options para evitar el error "should be a sequence"
            FACE_ANALYSER = insightface.app.FaceAnalysis(
                name='buffalo_l', 
                providers=providers,
                root=os.path.dirname(os.path.dirname(__file__))
            )
            ctx_id = 0 if use_cuda else -1
            FACE_ANALYSER.prepare(ctx_id=ctx_id, det_size=(320, 320))
            print(f"[SUCCESS] [OK] FaceAnalysis inicializado con {providers[0]} (det_size=320x320)")
            
        except Exception as e:
            print(f"[ERROR] Falló inicialización: {e}")
            print(f"[INFO] Intentando con CPU...")
            # Fallback a CPU si falla
            FACE_ANALYSER = insightface.app.FaceAnalysis(
                name='buffalo_l', 
                providers=['CPUExecutionProvider'],
                root=os.path.dirname(os.path.dirname(__file__))
            )
            FACE_ANALYSER.prepare(ctx_id=-1, det_size=(320, 320))
            print(f"[INFO] FaceAnalysis usando CPU por fallback (det_size=320x320)")
    
    return FACE_ANALYSER  

def get_face_single(img_data):  
    face = get_face_analyser().get(img_data)  
    try:  
        return sorted(face, key=lambda x: x.bbox[0])[0]  
    except IndexError:  
        return None  
      
def get_face_many(img_data):  
    try:  
        return get_face_analyser().get(img_data)  
    except IndexError:  
        return None

