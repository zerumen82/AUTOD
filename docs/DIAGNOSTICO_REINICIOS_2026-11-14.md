# Por qué se reinició el PC — 2026-11-14

Revisión de los registros de Windows (System, Application, nvlddmkm, WHEA,
volúmenes) y del estado de los discos. No es un cuelgue de AutoAuto: el
sistema se congeló con la GPU a plena carga y la única anomalía previa
registrada por el hardware es el disco USB de 2 TB (unidad `E:`).

## 1. Qué dicen los registros

| Hora local | Evento | Contenido |
|---|---|---|
| 14/11 20:34:46 | 6008 | «El cierre anterior … resultó inesperado» |
| 14/11 20:58:18 | **41** Kernel-Power (crítico) | Reinicio sin apagado limpio. `BugcheckCode=0` |
| 14/11 21:33:22 | **51** disk ×17 en 1 s | «Error detectado en el dispositivo `\Device\Harddisk2\DR2` durante una operación de paginación» |
| 14/11 21:33:36 | 98 Ntfs | La unidad `E:` reaparece con OTRO número de volumen (`HarddiskVolume8`) → el disco se desconectó y volvió a enumerarse |
| 14/11 21:38:27 | 6008 | «El cierre anterior de las 21:38:27 resultó inesperado» |
| 14/11 21:42:58 | **41** Kernel-Power (crítico) | Segundo reinicio sin apagado limpio. `BugcheckCode=0` |

Sin BSOD y sin volcado: `BugcheckCode=0`, `PowerButtonTimestamp=0`,
`C:\Windows\Minidump` solo tiene el volcado del 22/10 y `MEMORY.DMP` no existe.
`WHEA-Logger` está vacío (ningún error de máquina del procesador/PCIe) y no hay
TDR de pantalla (evento 4101 `Display`). Es decir: **no fue un pantallazo azul,
fue un cuelgue duro o un corte de alimentación**.

Contexto de las dos caídas: en ambas había un render del Image Editor en
marcha (`logs/p1_background_20261114*.log` a las 20:34 y
`logs/p1_matrix_rest_20261114.log` a las 21:38, los dos cortados a mitad de
pasada), y el registro de sistema queda mudo durante las dos horas previas al
primer cuelgue (18:31 → 20:34:46): la máquina dejó de escribir antes de caer.

## 2. El disco USB (unidad `E:`) es el único componente que falla

El mismo evento 51 aparece siempre en el mismo disco y en ráfagas:

| Día | Dispositivo | Errores |
|---|---|---|
| 08/10/2026 | Harddisk2 (DR2 y DR5) | 884 + 16 |
| 13/10/2026 | Harddisk2 (DR2, DR6) | 17 + 2 |
| 14/10/2026 | Harddisk2 (DR3) | 16 |
| 01/11/2026 | Harddisk2 (DR2) | 24 |
| 14/11/2026 | Harddisk2 (DR2) | 17 |

El dispositivo es `ST2000DM008-2UB102` de 2 TB dentro de una caja USB
(JMicron, `USB\VID_152D&PID_2338`, `USBSTOR`), unidad `E:`. Cada ráfaga
cambia el número `DRn`/`HarddiskVolumen`: el puente se cae y vuelve. Un
dispositivo de almacenamiento que desaparece bajo E/S paginada bloquea al
kernel y ese bloqueo es exactamente el patrón de las dos caídas del 14/11.

Los discos internos están sanos: `C:` (SSD CT480BX500) y `D:` (ST2000DM008
SATA, donde viven los 338 GB de modelos) reportan `Healthy / OK`. El archivo
de paginación está en `C:` (32 GB) y `D:` (5 GB), nunca en `E:`.

## 3. Cambios aplicados (software, reversibles)

1. **Disco duro: «Apagar el disco duro tras» = Nunca.** Estaba en 1200 s
   (20 min) incluso en el plan «Alto rendimiento». En una caja USB el
   spin-down/spin-up es el disparador clásico de la desconexión del puente y
   del bloqueo posterior. Aplicado con
   `powercfg -setacvalueindex SCHEME_CURRENT SUB_DISK DISKIDLE 0` (AC y DC).
2. Comprobado que ya estaban bien: suspensión selectiva de USB
   **deshabilitada**, ASPM de PCI Express **desactivado** (los dos son causa
   habitual de cuelgues con GPU), inicio rápido **desactivado**.
3. Volcado configurado para el próximo fallo: `CrashDumpEnabled=3` (kernel),
   `C:\WINDOWS\MEMORY.DMP`, `AutoReboot=0` (si hay bugcheck se queda en
   pantalla para poder leerlo) y sistema con espacio de sobra en `C:`.

## 4. Lo que queda en manos del usuario (hardware)

- **Desconectar la caja USB de 2 TB (`E:`) y probar unos días.** Si el PC deja
  de colgarse, el culpable es esa caja o su fuente/cable; si vuelve a caer,
  el siguiente sospechoso es la fuente del equipo (dos cuelgues con la GPU a
  tope y sin rastro de software).
- Cambiar cable y puerto USB (usar uno trasero, directo a la placa) y, si
  sigue, mover el disco a otra caja o probarlo dentro del equipo por SATA.
- Si se quiere descartar la fuente sin desmontar nada: limitar la GPU a
  ~70 % (`nvidia-smi -pl 140`) y comprobar si dejan de producirse los
  cuelgues.

## 5. Estado del entorno en el momento de esta revisión

- `Alto rendimiento`, `RTX 3060 Ti` a 44 °C / 20 W en reposo, límite 200 W,
  driver 32.0.16.2124 (07/11/2026). Sin errores ECC ni WHEA.
- **Dos ComfyUI a la vez en el puerto 8188**: PID 1736 con el Python del
  sistema (3.10, `AppData\Local\Programs\Python\Python310`) y PID 15336 con
  el venv del proyecto. El que escucha es el del sistema, así que el cliente
  del editor lo rechaza con «Puerto 8188 ocupado por un proceso ajeno (no es
  nuestro ComfyUI)». Había un render LongCat en cola en ese momento: no se
  tocó ninguno de los dos procesos. Para dejar el entorno limpio cuando no
  haya render en marcha:
  `CERRAR_TODO_Y_ABRIR_UNO.bat` (mata 8188, borra `logs/active_server.json` y
  abre una sola ventana con el venv).
- `logs/active_server.json` quedó apuntando a v5.1355/PID 17232 (muerto):
  lo reescribe la UI al arrancar, pero mientras tanto el lanzador puede leer
  una versión vieja. Borrarlo antes de abrir si vuelve a pasar.
