import insightface  
import roop.globals  
import os
import onnxruntime as ort

FACE_ANALYSER = None  

def get_face_analyser():  
    global FACE_ANALYSER  
    if FACE_ANALYSER is None:  
        # Asegurar que los modelos se carguen desde la ruta correcta
        models_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'buffalo_l')
        
        # VERIFICAR CUÁNTOS PROVEEDORES ESTÁN REALMENTE DISPONIBLES
        available_providers = ort.get_available_providers()
        print(f"[INFO] Proveedores ONNX Runtime disponibles: {available_providers}")
        
        # FORZAR CUDAExecutionProvider si está disponible
        if 'CUDAExecutionProvider' in available_providers:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print(f"[INFO] [OK] FORZANDO CUDA PARA FACE DETECTION")
        else:
            providers = ['CPUExecutionProvider']
            print(f"[INFO] ⚠️ CUDA no disponible, usando CPU")
        
        print(f"[INFO] Inicializando FaceAnalysis con providers: {providers}")
        
        try:
            # SIN provider_options para evitar el error "should be a sequence"
            FACE_ANALYSER = insightface.app.FaceAnalysis(
                name='buffalo_l', 
                providers=providers,
                root=os.path.dirname(os.path.dirname(__file__))
            )
            FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
            print(f"[SUCCESS] [OK] FaceAnalysis inicializado con {providers[0]}")
            
            # Verificar que realmente esté usando CUDA
            if hasattr(FACE_ANALYSER, 'models') and FACE_ANALYSER.models:
                cuda_count = 0
                for model_name, model in FACE_ANALYSER.models.items():
                    if hasattr(model, 'session') and hasattr(model.session, 'get_providers'):
                        model_providers = model.session.get_providers()
                        if 'CUDAExecutionProvider' in model_providers:
                            cuda_count += 1
                        print(f"  - {model_name}: {model_providers}")
                
                if cuda_count > 0:
                    print(f"[SUCCESS] [OK] {cuda_count} modelos usando CUDA")
                else:
                    print(f"[WARNING] ⚠️ Todos los modelos usando CPU")
            
        except Exception as e:
            print(f"[ERROR] Falló inicialización con CUDA: {e}")
            print(f"[INFO] Intentando con CPU...")
            # Fallback a CPU si CUDA falla
            FACE_ANALYSER = insightface.app.FaceAnalysis(
                name='buffalo_l', 
                providers=['CPUExecutionProvider'],
                root=os.path.dirname(os.path.dirname(__file__))
            )
            FACE_ANALYSER.prepare(ctx_id=0, det_size=(640, 640))
            print(f"[INFO] FaceAnalysis usando CPU por fallback")
    
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

