---
name: workflow-segmenter
description: Divide tareas complejas en pasos atómicos y verificables. Úsalo para planificar ejecuciones multi-etapa, reducir errores y facilitar la delegación a sub-agentes sin perder el contexto.
---

# Workflow Segmenter

Esta habilidad optimiza la ejecución de tareas complejas descomponiéndolas en un flujo de trabajo estructurado.

## Principios de Segmentación

### 1. El Método Atómico
Nunca realices más de una modificación lógica por paso. 
- **Paso Malo**: "Arreglar el bug de selección y añadir el botón de abrir carpeta."
- **Paso Bueno**: 
  1. Identificar por qué falla el clic de selección.
  2. Implementar corrección del clic.
  3. Verificar corrección.
  4. Añadir componente de botón en UI.
  5. Conectar evento del botón.

### 2. Estructura de Planificación
Antes de actuar en una tarea de >3 pasos, genera un "Checklist de Ejecución":

```markdown
### 📋 Plan de Ejecución
- [ ] **Etapa 1: Investigación**: Listar archivos afectados.
- [ ] **Etapa 2: Implementación**: Aplicar cambios quirúrgicos.
- [ ] **Etapa 3: Verificación**: Ejecutar tests o logs de comprobación.
```

### 3. Puntos de Control (Checkpoints)
Al finalizar cada paso del checklist, detente un milisegundo mentalmente para validar:
- ¿El cambio introducido rompe alguna dependencia detectada en el paso 1?
- ¿El output coincide con lo esperado por el usuario?

## Uso con Sub-agentes
Cuando delegues una tarea a un sub-agente (como `codebase_investigator`), pásale solo el segmento actual del plan. No lo abrumes con todo el proyecto si solo necesita arreglar una función.
