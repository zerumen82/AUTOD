import cv2
import os
import numpy as np
from PIL import Image, ImageEnhance

def create_video_from_images(output_path, image_dir, fps=5, duration=5):
    """Crea un video a partir de imรกgenes en un directorio"""
    
    # Obtener imรกgenes
    images = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            img_path = os.path.join(image_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                # Redimensionar para que todas tengan el mismo tamaรฑo
                img = cv2.resize(img, (640, 480))
                images.append(img)
    
    if len(images) == 0:
        print("No se encontraron imรกgenes")
        return
    
    # Crear video
    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Escribir imรกgenes en el video
    for img in images:
        for _ in range(int(fps * duration / len(images))):
            video.write(img)
    
    video.release()
    print(f"Video guardado en: {output_path}")

def create_fade_transition_video(output_path, image_dir, fps=10):
    """Crea un video con transiciones de desvanecimiento entre imรกgenes"""
    
    images = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            img_path = os.path.join(image_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (640, 480))
                images.append(img)
    
    if len(images) < 2:
        print("Se necesitan al menos 2 imรกgenes para transiciones")
        return
    
    height, width, _ = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Crear transiciones de desvanecimiento
    for i in range(len(images) - 1):
        img1 = images[i]
        img2 = images[i + 1]
        
        # Mostrar primera imagen
        for _ in range(fps):
            video.write(img1)
        
        # Transiciรณn de desvanecimiento
        for alpha in np.linspace(0, 1, fps):
            blended = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
            video.write(blended)
    
    # Mostrar รบltima imagen
    for _ in range(fps * 2):
        video.write(images[-1])
    
    video.release()
    print(f"Video con transiciones guardado en: {output_path}")

def create_pil_effects_video(output_path, image_dir, fps=10):
    """Crea un video con efectos usando PIL (brillo, contraste, etc.)"""
    
    images = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path)
            img = img.resize((640, 480))
            images.append(img)
    
    if len(images) == 0:
        print("No se encontraron imรกgenes")
        return
    
    height, width = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Aplicar efectos a cada imagen
    for img in images:
        # Convertir a array de numpy para OpenCV
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Efecto 1: Normal
        for _ in range(fps):
            video.write(img_cv)
        
        # Efecto 2: Aumentar brillo
        enhancer = ImageEnhance.Brightness(img)
        bright_img = enhancer.enhance(1.5)
        bright_cv = cv2.cvtColor(np.array(bright_img), cv2.COLOR_RGB2BGR)
        for _ in range(fps):
            video.write(bright_cv)
        
        # Efecto 3: Aumentar contraste
        enhancer = ImageEnhance.Contrast(img)
        contrast_img = enhancer.enhance(1.5)
        contrast_cv = cv2.cvtColor(np.array(contrast_img), cv2.COLOR_RGB2BGR)
        for _ in range(fps):
            video.write(contrast_cv)
        
        # Efecto 4: Combinar brillo y contraste
        enhancer = ImageEnhance.Brightness(img)
        temp_img = enhancer.enhance(1.3)
        enhancer = ImageEnhance.Contrast(temp_img)
        final_img = enhancer.enhance(1.3)
        final_cv = cv2.cvtColor(np.array(final_img), cv2.COLOR_RGB2BGR)
        for _ in range(fps):
            video.write(final_cv)
    
    video.release()
    print(f"Video con efectos PIL guardado en: {output_path}")

def create_gif_from_images(output_path, image_dir, duration=2):
    """Crea un GIF animado a partir de imรกgenes"""
    
    images = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path)
            img = img.resize((640, 480))
            images.append(img)
    
    if len(images) == 0:
        print("No se encontraron imรกgenes")
        return
    
    # Crear GIF
    images[0].save(output_path, save_all=True, append_images=images[1:], 
                   optimize=False, duration=duration*len(images), loop=0)
    print(f"GIF guardado en: {output_path}")

def create_imagine_style_video(output_path, image_dir, fps=10):
    """Crea un video estilo 'Imagine' como en Grok"""
    
    images = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path)
            img = img.resize((640, 480))
            images.append(img)
    
    if len(images) == 0:
        print("No se encontraron imรกgenes")
        return
    
    height, width = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Efecto Imagine: zoom lento y movimiento sutil
    for img in images:
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Mostrar imagen normal
        for _ in range(fps):
            video.write(img_cv)
        
        # Efecto de zoom lento
        for scale in np.linspace(1.0, 1.1, fps):
            h, w = img_cv.shape[:2]
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img_cv, (new_w, new_h))
            x_start = (new_w - w) // 2
            y_start = (new_h - h) // 2
            cropped = resized[y_start:y_start+h, x_start:x_start+w]
            video.write(cropped)
        
        # Efecto de movimiento horizontal sutil
        for offset in range(-10, 10):
            M = np.float32([[1, 0, offset], [0, 1, 0]])
            shifted = cv2.warpAffine(img_cv, M, (w, h))
            video.write(shifted)
    
    video.release()
    print(f"Video estilo Imagine guardado en: {output_path}")

def create_animacion_movimiento(output_path, image_dir, fps=10, duration_seconds=6):
    """Crea una animaciรณn corta con movimiento a partir de una imagen"""
    
    images = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path)
            img = img.resize((640, 480))
            images.append(img)
    
    if len(images) == 0:
        print("No se encontraron imรกgenes")
        return
    
    height, width = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Crear animaciรณn de movimiento con la primera imagen
    img = images[0]
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Efecto de movimiento: desplazamiento horizontal
    for frame in range(fps * duration_seconds):
        # Calcular desplazamiento basado en el tiempo
        offset = int((frame / (fps * duration_seconds)) * width * 0.5)
        
        # Crear imagen desplazada
        M = np.float32([[1, 0, offset], [0, 1, 0]])
        shifted = cv2.warpAffine(img_cv, M, (width, height))
        
        # Agregar efecto de zoom sutil
        scale = 1.0 + (frame / (fps * duration_seconds)) * 0.1
        h, w = shifted.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(shifted, (new_w, new_h))
        x_start = (new_w - w) // 2
        y_start = (new_h - h) // 2
        final = resized[y_start:y_start+h, x_start:x_start+w]
        
        video.write(final)
    
    video.release()
    print(f"Animaciรณn de movimiento guardada en: {output_path}")

def create_animacion_bailando(output_path, image_dir, fps=10, duration_seconds=6):
    """Crea una animaciรณn de personas bailando a partir de una imagen"""
    
    images = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path)
            img = img.resize((640, 480))
            images.append(img)
    
    if len(images) == 0:
        print("No se encontraron imรกgenes")
        return
    
    height, width = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Crear animaciรณn de baile con la primera imagen
    img = images[0]
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Efectos de baile: movimiento de brazos y piernas
    for frame in range(fps * duration_seconds):
        # Calcular รกngulo de movimiento basado en el tiempo
        angle = (frame / (fps * duration_seconds)) * 360
        
        # Rotar imagen para simular movimiento
        M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
        rotated = cv2.warpAffine(img_cv, M, (width, height))
        
        # Agregar efecto de escala pulsante
        scale = 1.0 + 0.1 * np.sin(2 * np.pi * frame / fps)
        h, w = rotated.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(rotated, (new_w, new_h))
        x_start = (new_w - w) // 2
        y_start = (new_h - h) // 2
        final = resized[y_start:y_start+h, x_start:x_start+w]
        
        # Agregar efecto de brillo pulsante
        hsv = cv2.cvtColor(final, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.add(hsv[:, :, 2], np.sin(2 * np.pi * frame / fps) * 30)
        final = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        video.write(final)
    
    video.release()
    print(f"Animaciรณn de baile guardada en: {output_path}")

def create_animacion_integrada_ui(output_path, image_dir, fps=10, duration_seconds=6):
    """Crea una animaciรณn con interfaz de usuario integrada"""
    
    images = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path)
            img = img.resize((640, 480))
            images.append(img)
    
    if len(images) == 0:
        print("No se encontraron imรกgenes")
        return
    
    height, width = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Crear animaciรณn con UI integrada
    img = images[0]
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Fondo con gradiente
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        gradient[i] = [int(50 + i * 0.5), int(50 + i * 0.5), int(50 + i * 0.5)]
    
    # Superponer imagen sobre gradiente
    alpha = 0.7
    blended = cv2.addWeighted(img_cv, alpha, gradient, 1 - alpha, 0)
    
    # Agregar elementos de UI
    for frame in range(fps * duration_seconds):
        # Limpiar canvas
        canvas = blended.copy()
        
        # Efecto de movimiento
        offset = int((frame / (fps * duration_seconds)) * width * 0.3)
        M = np.float32([[1, 0, offset], [0, 1, 0]])
        shifted = cv2.warpAffine(canvas, M, (width, height))
        
        # Efecto de brillo pulsante
        hsv = cv2.cvtColor(shifted, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.add(hsv[:, :, 2], np.sin(2 * np.pi * frame / fps) * 20)
        final = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Agregar controles de UI (simulados)
        cv2.rectangle(final, (10, 10), (100, 40), (255, 255, 255), -1)
        cv2.putText(final, "Play", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.rectangle(final, (110, 10), (200, 40), (255, 255, 255), -1)
        cv2.putText(final, "Pause", (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Agregar elementos interactivos
        cv2.circle(final, (int(width/2), int(height/2)), 50, (0, 255, 0), 2)
        cv2.putText(final, "Hacer clic aquรญ", (int(width/2)-40, int(height/2)+60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        video.write(final)
    
    video.release()
    print(f"Animaciรณn con UI integrada guardada en: {output_path}")

def create_animacion_interactiva(output_path, image_dir, fps=10, duration_seconds=6):
    """Crea una animaciรณn interactiva con respuesta a acciones"""
    
    images = []
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            img_path = os.path.join(image_dir, filename)
            img = Image.open(img_path)
            img = img.resize((640, 480))
            images.append(img)
    
    if len(images) == 0:
        print("No se encontraron imรกgenes")
        return
    
    height, width = 480, 640
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Crear animaciรณn interactiva
    img = images[0]
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    
    # Fondo con gradiente
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        gradient[i] = [int(50 + i * 0.5), int(50 + i * 0.5), int(50 + i * 0.5)]
    
    # Superponer imagen sobre gradiente
    alpha = 0.7
    blended = cv2.addWeighted(img_cv, alpha, gradient, 1 - alpha, 0)
    
    # Agregar elementos interactivos
    for frame in range(fps * duration_seconds):
        # Limpiar canvas
        canvas = blended.copy()
        
        # Efecto de movimiento
        offset = int((frame / (fps * duration_seconds)) * width * 0.3)
        M = np.float32([[1, 0, offset], [0, 1, 0]])
        shifted = cv2.warpAffine(canvas, M, (width, height))
        
        # Efecto de brillo pulsante
        hsv = cv2.cvtColor(shifted, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.add(hsv[:, :, 2], np.sin(2 * np.pi * frame / fps) * 20)
        final = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Agregar controles interactivos
        cv2.rectangle(final, (10, 10), (100, 40), (255, 255, 255), -1)
        cv2.putText(final, "Hacer clic", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.rectangle(final, (110, 10), (200, 40), (255, 255, 255), -1)
        cv2.putText(final, "Arrastrar", (120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Simular respuesta a clic
        if frame > fps * 2 and frame < fps * 4:
            cv2.circle(final, (int(width/2), int(height/2)), 30, (0, 255, 0), -1)
            cv2.putText(final, "ยกClic detectado!", (int(width/2)-50, int(height/2)+40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        
        # Simular respuesta a arrastre
        if frame > fps * 4 and frame < fps * 6:
            cv2.rectangle(final, (int(width/2)-40, int(height/2)-40), 
                         (int(width/2)+40, int(height/2)+40), (255, 0, 0), -1)
            cv2.putText(final, "ยกArrastre detectado!", (int(width/2)-60, int(height/2)+60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        video.write(final)
    
    video.release()
    print(f"Animaciรณn interactiva guardada en: {output_path}")

if __name__ == "__main__":
    # Crear diferentes tipos de videos
    create_video_from_images("video_simple.mp4", "testdata")
    create_fade_transition_video("video_transiciones.mp4", "testdata")
    create_pil_effects_video("video_efectos.mp4", "testdata")
    create_gif_from_images("animacion.gif", "testdata")
    create_imagine_style_video("video_imagine.mp4", "testdata")
    create_animacion_movimiento("animacion_movimiento.mp4", "testdata")
    create_animacion_bailando("animacion_bailando.mp4", "testdata")
    create_animacion_integrada_ui("animacion_ui.mp4", "testdata")
    create_animacion_interactiva("animacion_interactiva.mp4", "testdata")
    print("\n=== Videos y GIF creados exitosamente ===")
    print("1. video_simple.mp4 - Video bรกsico con todas las imรกgenes")
    print("2. video_transiciones.mp4 - Video con transiciones de desvanecimiento")
    print("3. video_efectos.mp4 - Video con efectos de brillo y contraste")
    print("4. animacion.gif - GIF animado con todas las imรกgenes")
    print("5. video_imagine.mp4 - Video estilo 'Imagine' como en Grok")
    print("6. animacion_movimiento.mp4 - Animaciรณn corta con movimiento")
    print("7. animacion_bailando.mp4 - Animaciรณn de personas bailando")
    print("8. animacion_ui.mp4 - Animaciรณn con interfaz de usuario integrada")
    print("9. animacion_interactiva.mp4 - Animaciรณn interactiva con respuesta a acciones")