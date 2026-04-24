# 📋 Checklist de Mejoras: FaceSwap e Image Editor

Edita este archivo añadiendo tus precisiones en cada punto. Cuando termines, avísame en el chat.

---

## 👤 Pestaña FaceSwap (Optimización y UX)

### 1. Refactorización de `faceswap_tab.py`
* **Idea:** Dividir el archivo de +4600 líneas en módulos (ui, logic, processing).
* **[MIS PRECISIONES]:** 
	OK HAZLO PERO DE FORMA QUIRURGICA SIN ROMPEER NADA DE LO QUE FUNCIONA

### 2. Propagación Inteligente de Caras
* **Idea:** Auto-selección por embedding facial en todo el lote tras la primera selección.
* **[MIS PRECISIONES]:** 
	OK PERO SOLO EN EL MODO SELECTED FACES FRAME.PORQUE SON VIDEOS ENTOINCES SII QUIE INTERESA YA QUE LE DARIA MAS ESTRIBILIADA AL SWAP.
	NO EN EL MODO SELECTED FACES YA QUE SON IMAGENES Y LA CARA DSE ORIGEN PUEDE NO SER LA MISMA PERSONA Y TAMBIOEN EN EL DESTINO PUEDEN SER PERSONAS DIFERENTES. SIQUUIERFES UN ANALISIS IMAGEN POR IMAGEN DE ORIGEN QUE 
	MAS SE PAREZCA PERO ESO IWAL REALENTIZA EL SWAP.-

### 3. Filtros de Galería
* **Idea:** Filtros rápidos por género o tamaño de rostro en fotos grupales.
* **[MIS PRECISIONES]:** 
	OK- PERO QUE NO SEA MUY INTRUSIVOI EN LA UI	

### 4. Preview de Enhancer "Ojo"
* **Idea:** Botón para ver un "crop" rápido del antes/después del enhancer.
* **[MIS PRECISIONES]:** 
	OK-PERO NO MUIY DISRUPTIVO EN LA UI

### 5. Detección de Oclusiones
* **Idea:** Evitar que el swap tape pelo o manos que cruzan la cara.
* **[MIS PRECISIONES]:** 
	OK

QUIERO AñADIR MEJORAS:

---- CUANDEOO SE PULSE USE THIS FRAME DETECTA LAS CARAS Y TENGO QUE MOVIENDO UN SLIDER Y LUEGO PULSAR USE THIS FACE O ALLGOO ASI PARA QUE ME SELECIONE UNA CARA DERSTINO. ME GUSTARIA PODER DAR UN CLICK Y SELECIONAR LA CARA DESTINO
---- EN LA GALERIA DE CARAS DESTINO PARA BORRAR UNA DE LAS CARAS ELEGIDAS. TENGO QUE SELECCIONA R QUE HBRA OTRA SUBGALERIA Y DE AHII PULSO REMOVE SELECTYED FACE. ES ENGORROSO.
-----  EN CUANDO SUBO MP4 OSEA VIDEOS Y DEBO SELECCIONAR CARAS DE DEWSTINO. ASLO SER UN VIDEO SOLO COGE EL PRIMER FRAME Y SINMO ESTYA LA CARA NO PUEDO HACERLO PORQUE EL SLIDER DE FRAMES PASA AL SIGGUIENTE VIDEO YA QUE INTERPRETA VIDEOS COMO FRAMES. ME GUSTARA PODER AVANZAR EN EL VIDEO Y ELEGIIR LA CFARA DE DESTINO Y QUE SIGA FUNCIONANDO EN EL CASO DE IMAGENES DE LA MANERA QUE FUNCIONA AHORA.
----- EL RESULTADO ES BASTANTE BUENO. PEROO AUN FRALLA EN PERFILES, O EN CARAS BOCA ABAJO. ADEMAS ME GUSTARIA QUE POR DEFERCTO ESTE PUESTO QUE SE PAREZCA LO MAXIMO MAXIMO MAXIMO A LA CARA DDE ORIGEN Y QUE LA CALIDAD SEA BU8ENA.
----- EN EL MODO SELECTED FACES FRAME. CUANDO ES PRIMER PLANO O CARAS GRANDES AUN SE NOTA CAMBIOS Y CIERTOS PARPADEOS.

---

## 🎨 Pestaña Image Editor (Funcionalidad y Control)

### 6. Canvas de Máscaras (Inpaint)
* **Idea:** Pintar a mano el área a modificar sobre la imagen.
* **[MIS PRECISIONES]:** 
	OK, A ME GUSTARIA QUE FUNCIUONASE COMO GROK IMAGINE -POR EJEMPLO: SUBO UNA FOTO MIA Y LE DIGO EN EL PROOMPR. VISTELO DE PAYASO Y Y HAZ QUE SALTE SOOBRE UNA PIERNA.)- ENTONCES COGE MI IMAGEN Y ME CAMBIA SOLO A MI LA ROPA Y ME PO0NE EN ESA POSICION O MANNTIENE ELK RESTYO O MODIFICA PERO CON LA MISMA ESTETICA 

### 7. Resolutor de Parámetros (Backend)
* **Idea:** Ocultar o bloquear sliders que no funcionan según el motor (HART, Klein, etc).
* **[MIS PRECISIONES]:** 
	OK

### 8. Comparador de Cortina (Before/After)
* **Idea:** Slider visual interactivo para comparar original y resultado.
* **[MIS PRECISIONES]:** 
	OK

### 9. Prompt-to-Mask (CLIPSeg)
* **Idea:** Generación automática de máscara escribiendo el objeto (ej. "vestido").
* **[MIS PRECISIONES]:** 
	OK

### 10. Fijación de Identidad (Personaje)
* **Idea:** Usar la cara de FaceSwap como referencia para que no cambie al editar.
* **[MIS PRECISIONES]:** 
	OK

---CUALQUIER CAMBIO QUE NOS LLEVE AL ESCENARIO REPRESENTADO EN EL PUNTO PRIMERO  ESTUILO GROIKM IMAGINE

---

## 🤖 Mejoras de IA Generales

### 11. Análisis de Imagen Mejorado
* **Idea:** Detección de personas y sugerencia de prompts automáticos.
* **[MIS PRECISIONES]:** 
	OK

### 12. Cola de Trabajos
* **Idea:** Sistema para mandar varios edits y que se procesen en segundo plano.
* **[MIS PRECISIONES]:** 
        OK
---
## 💡 Otras ideas o notas generales
* **[ESCRIBE AQUÍ]:** 
