# -*- coding: utf-8 -*-
import sys
import os
from pathlib import Path

# Añadir directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

try:
    print("Iniciando prueba de SemanticIntentAnalyzer...")
    from roop.img_editor.nlp.semantic_analyzer import SemanticIntentAnalyzer
    
    # Intentar cargar con un timeout corto o simplemente verificar si falla
    analyzer = SemanticIntentAnalyzer()
    
    test_prompts = [
        "Pon a la chica de rodillas",
        "Cambia el color de la camisa a rojo",
        "Añade un hombre al fondo",
        "Retoca los ojos"
    ]
    
    for p in test_prompts:
        mag = analyzer.get_magnitude(p)
        target = analyzer.detect_target(p)
        print(f"Prompt: {p}")
        print(f"  Mag: {mag:.2f}")
        print(f"  Target: {target}")
        print("-" * 20)
        
    print("Prueba completada con éxito.")
except Exception as e:
    print(f"Error durante la prueba: {e}")
    import traceback
    traceback.print_exc()
