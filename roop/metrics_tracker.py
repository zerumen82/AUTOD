"""
metrics_tracker.py - Seguimiento de métricas en tiempo real para procesamiento de video

Proporciona:
- Progreso en tiempo real
- Tiempo restante estimado
- FPS procesados
- Contador de frames
"""

import time
from typing import Optional


class MetricsTracker:
    """
    Rastrea métricas de procesamiento de video en tiempo real.
    
    Uso:
        tracker = MetricsTracker(total_frames=300)
        tracker.update_frame_processed(frame_num=50)
        progress_html = tracker.get_progress_html()
    """
    
    def __init__(self, total_frames: int):
        """
        Inicializa el tracker.
        
        Args:
            total_frames: Número total de frames a procesar
        """
        self.total_frames = total_frames
        self.processed_frames = 0
        self.start_time = None
        self.last_update_time = None
        self.fps_history = []
        self.is_running = False
        
    def start(self):
        """Inicia el tracking"""
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.processed_frames = 0
        self.fps_history = []
        self.is_running = True
        
    def update_frame_processed(self, frame_num: Optional[int] = None):
        """
        Actualiza el contador de frames procesados.
        
        Args:
            frame_num: Número de frame actual (opcional, se auto-incrementa si es None)
        """
        if not self.is_running:
            self.start()
        
        current_time = time.time()
        
        # Incrementar frames procesados
        if frame_num is not None:
            self.processed_frames = frame_num
        else:
            self.processed_frames += 1
        
        # Calcular FPS instantáneo
        if self.last_update_time:
            time_diff = current_time - self.last_update_time
            if time_diff > 0:
                instant_fps = 1.0 / time_diff
                self.fps_history.append(instant_fps)
                
                # Mantener solo últimos 30 FPS para promedio móvil
                if len(self.fps_history) > 30:
                    self.fps_history.pop(0)
        
        self.last_update_time = current_time
        
    def get_progress_percent(self) -> float:
        """Obtiene porcentaje de progreso (0-100)"""
        if self.total_frames <= 0:
            return 0.0
        return min(100.0, (self.processed_frames / self.total_frames) * 100)
    
    def get_elapsed_time(self) -> float:
        """Obtiene tiempo transcurrido en segundos"""
        if not self.start_time:
            return 0.0
        return time.time() - self.start_time
    
    def get_remaining_time(self) -> float:
        """Obtiene tiempo restante estimado en segundos"""
        if self.processed_frames <= 0 or not self.start_time:
            return 0.0
        
        elapsed = self.get_elapsed_time()
        progress = self.processed_frames / self.total_frames
        
        if progress <= 0:
            return 0.0
        
        total_estimated = elapsed / progress
        remaining = total_estimated - elapsed
        
        return max(0.0, remaining)
    
    def get_fps(self) -> float:
        """Obtiene FPS promedio (últimos 30 frames)"""
        if not self.fps_history:
            return 0.0
        return sum(self.fps_history) / len(self.fps_history)
    
    def format_time(self, seconds: float) -> str:
        """Formatea segundos a HH:MM:SS o MM:SS"""
        if seconds <= 0:
            return "--:--"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def get_metrics_dict(self) -> dict:
        """
        Obtiene todas las métricas en diccionario.
        
        Returns:
            dict con:
                - progress_percent: 0-100
                - processed_frames: int
                - total_frames: int
                - fps: float
                - elapsed_time: float (segundos)
                - remaining_time: float (segundos)
                - remaining_time_formatted: str
        """
        return {
            'progress_percent': self.get_progress_percent(),
            'processed_frames': self.processed_frames,
            'total_frames': self.total_frames,
            'fps': self.get_fps(),
            'elapsed_time': self.get_elapsed_time(),
            'remaining_time': self.get_remaining_time(),
            'remaining_time_formatted': self.format_time(self.get_remaining_time())
        }
    
    def get_progress_html(self) -> str:
        """
        Genera HTML actualizado para mostrar métricas.
        
        Returns:
            str: HTML con las métricas actuales
        """
        metrics = self.get_metrics_dict()
        
        # Determinar color de progreso
        progress = metrics['progress_percent']
        if progress < 25:
            progress_color = "#3b82f6"  # Azul
        elif progress < 50:
            progress_color = "#06b6d4"  # Cyan
        elif progress < 75:
            progress_color = "#10b981"  # Verde
        else:
            progress_color = "#f59e0b"  # Naranja
        
        html = f"""
        <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); 
                    padding: 15px; border-radius: 10px; margin: 10px 0;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h3 style="color: #3b82f6; margin-top: 0; font-size: 16px;">
                📊 Métricas en Tiempo Real
            </h3>
            <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px;">
                <div style="text-align: center;">
                    <div style="color: #94a3b8; font-size: 12px; margin-bottom: 5px;">Progreso</div>
                    <div style="color: {progress_color}; font-size: 24px; font-weight: bold;">
                        {progress:.1f}%
                    </div>
                </div>
                <div style="text-align: center;">
                    <div style="color: #94a3b8; font-size: 12px; margin-bottom: 5px;">Tiempo Restante</div>
                    <div style="color: #10b981; font-size: 24px; font-weight: bold;">
                        {metrics['remaining_time_formatted']}
                    </div>
                </div>
                <div style="text-align: center;">
                    <div style="color: #94a3b8; font-size: 12px; margin-bottom: 5px;">FPS</div>
                    <div style="color: #f59e0b; font-size: 24px; font-weight: bold;">
                        {metrics['fps']:.1f}
                    </div>
                </div>
                <div style="text-align: center;">
                    <div style="color: #94a3b8; font-size: 12px; margin-bottom: 5px;">Frames</div>
                    <div style="color: #8b5cf6; font-size: 24px; font-weight: bold;">
                        {metrics['processed_frames']}/{metrics['total_frames']}
                    </div>
                </div>
            </div>
            <!-- Barra de progreso -->
            <div style="margin-top: 15px; background: #334155; border-radius: 5px; height: 8px; overflow: hidden;">
                <div style="background: linear-gradient(90deg, {progress_color} 0%, #3b82f6 100%); 
                            height: 100%; width: {progress}%; transition: width 0.3s;"></div>
            </div>
        </div>
        """
        
        return html
    
    def stop(self):
        """Detiene el tracking"""
        self.is_running = False


# Instancia global para usar en ProcessMgr
_current_tracker: Optional[MetricsTracker] = None


def get_current_tracker() -> Optional[MetricsTracker]:
    """Obtiene el tracker actual"""
    return _current_tracker


def set_current_tracker(tracker: MetricsTracker):
    """Establece el tracker actual"""
    global _current_tracker
    _current_tracker = tracker


def update_metrics(frame_num: int = None) -> str:
    """
    Actualiza métricas y retorna HTML actualizado.
    
    Args:
        frame_num: Número de frame actual
        
    Returns:
        str: HTML actualizado con métricas
    """
    if _current_tracker is None:
        return ""
    
    _current_tracker.update_frame_processed(frame_num)
    return _current_tracker.get_progress_html()
