#!/usr/bin/env python3
"""
install_mediapipe.py
Instala MediaPipe para mejor detección de boca (468 landmarks)
"""

import subprocess
import sys

def install_mediapipe():
    """Instala MediaPipe con verificación"""
    print("=" * 60)
    print("   INSTALACIÓN - MEDIAPIPE FACE MESH")
    print("=" * 60)
    print("\n📦 MediaPipe proporciona 468 landmarks faciales")
    print("   para detección precisa de boca abierta")
    print("\n" + "=" * 60)
    
    try:
        # Verificar si ya está instalado
        import mediapipe
        print(f"\n✅ MediaPipe ya está instalado: v{mediapipe.__version__}")
        return 0
        
    except ImportError:
        print("\n⏳ Instalando MediaPipe...")
        
        try:
            subprocess.check_call([
                sys.executable, 
                "-m", 
                "pip", 
                "install",
                "mediapipe>=0.10.0",
                "--upgrade"
            ])
            
            print("\n✅ MediaPipe instalado exitosamente")
            
            # Verificar instalación
            import mediapipe
            print(f"   Versión: {mediapipe.__version__}")
            
            print("\n" + "=" * 60)
            print("✅ ¡INSTALACIÓN COMPLETADA!")
            print("=" * 60)
            print("\n💡 Ahora puedes usar detección de boca con 468 landmarks")
            print("   en la configuración de Face Swap")
            
            return 0
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Error instalando: {e}")
            print("\nIntenta manualmente:")
            print("   pip install mediapipe")
            return 1
        except Exception as e:
            print(f"\n❌ Error inesperado: {e}")
            return 1

if __name__ == "__main__":
    sys.exit(install_mediapipe())
