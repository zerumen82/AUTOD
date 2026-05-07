---
name: token-optimizer
description: Reglas de oro para minimizar el consumo de tokens y maximizar la eficiencia del contexto. Úsalo siempre que trabajes en repositorios grandes o con tareas que requieran múltiples lecturas de archivos.
---

# Token Optimizer

Maximiza la "señal" y minimiza el "ruido" en el contexto de la IA.

## Directrices de Búsqueda y Lectura

### 1. Búsqueda Quirúrgica (Grep)
No busques términos genéricos. Usa `grep_search` con patrones precisos y límites de resultados.
- **Mal**: `grep_search(pattern="face")`
- **Bien**: `grep_search(pattern="def on_face_selection_click", include_pattern="ui/tabs/faceswap/*.py")`

### 2. Lectura Segmentada (read_file)
NUNCA leas archivos de más de 100 líneas por completo a menos que sea estrictamente necesario para entender la arquitectura total.
- Usa `start_line` y `end_line` para leer solo el bloque de código relevante.
- Si necesitas contexto, lee 20 líneas antes y después del punto de interés.

### 3. Evitar el "Efecto Eco"
No repitas resúmenes extensos de lo que ya hiciste a menos que el usuario lo pida. El historial ya contiene tus acciones.
- **Mal**: "He modificado el archivo X para añadir Y, luego he ido a Z y he hecho W, y finalmente..."
- **Bien**: "He corregido el bug de selección en `events.py`. ¿Deseas probar el cambio?"

### 4. Gestión de Memoria (Project Memory)
Usa `save_memory` con scope `project` para persistir hechos arquitectónicos que descubras. Esto evita tener que volver a investigar lo mismo en la siguiente sesión de chat.

## Reglas para Sub-agentes
Al invocar un sub-agente:
- **No pases archivos enteros**: Solo rutas y nombres de símbolos.
- **Especifica el objetivo**: "Investiga por qué la variable X es None en la línea 45 de core.py" es mejor que "Arregla este archivo".
