import cv2
import os
from matplotlib import pyplot as plt

def check_test_images():
    """Verifica visualmente las imágenes de testdata"""
    
    test_dir = "testdata"
    if not os.path.exists(test_dir):
        print(f"Directorio {test_dir} no encontrado")
        return
    
    print(f"=== Verificando imágenes en {test_dir} ===")
    
    for filename in os.listdir(test_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(test_dir, filename)
            print(f"\nAnalizando: {filename}")
            
            # Leer imagen
            img = cv2.imread(img_path)
            if img is None:
                print("  No se pudo leer la imagen")
                continue
                
            print(f"  Dimensiones: {img.shape[1]}x{img.shape[0]}")
            print(f"  Canal: {img.shape[2] if len(img.shape) == 3 else 1}")
            
            # Mostrar imagen
            plt.figure(figsize=(6, 6))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title(filename)
            plt.axis('off')
            plt.show()
            
            # Probar detección de caras con OpenCV Haar Cascades (simple)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                print(f"  Caras detectadas: {len(faces)}")
                # Dibujar rectángulos
                img_with_faces = img.copy()
                for (x, y, w, h) in faces:
                    cv2.rectangle(img_with_faces, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                plt.figure(figsize=(6, 6))
                plt.imshow(cv2.cvtColor(img_with_faces, cv2.COLOR_BGR2RGB))
                plt.title(f"{filename} - {len(faces)} cara(s) detectada(s)")
                plt.axis('off')
                plt.show()
            else:
                print("  No se detectaron caras con Haar Cascades")

if __name__ == "__main__":
    check_test_images()
