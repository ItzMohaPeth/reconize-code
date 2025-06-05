import cv2
import time
import numpy as np
from typing import Optional, Tuple, List, Dict, Any, Callable

class VideoProcessor:
    """
    Classe pour gérer le traitement des flux vidéo.
    """
    def __init__(self, source: str = "0", output_path: Optional[str] = None, 
                 resolution: Optional[Tuple[int, int]] = None, fps: Optional[int] = None):
        """
        Initialise le processeur vidéo.
        
        Args:
            source: Chemin vers le fichier vidéo ou ID de la caméra (0 pour webcam par défaut)
            output_path: Chemin pour sauvegarder la vidéo traitée (None pour ne pas sauvegarder)
            resolution: Tuple (width, height) pour redimensionner les frames
            fps: Frames par seconde pour la vidéo de sortie
        """
        self.source = source
        self.output_path = output_path
        self.resolution = resolution
        self.fps = fps
        self.cap = None
        self.writer = None
        self.frame_count = 0
        self.start_time = None
        self.processing_fps = 0
        
    def __enter__(self):
        """Permet l'utilisation du context manager (with)"""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Ferme les ressources à la sortie du context manager"""
        self.release()
        
    def start(self):
        """Initialise la capture vidéo et le writer si nécessaire"""
        # Initialiser la capture vidéo
        if self.source.isdigit():
            self.cap = cv2.VideoCapture(int(self.source))
        else:
            self.cap = cv2.VideoCapture(self.source)
            
        if not self.cap.isOpened():
            raise ValueError(f"Impossible d'ouvrir la source vidéo: {self.source}")
            
        # Obtenir les propriétés de la vidéo
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.original_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        # Configurer la résolution si spécifiée
        if self.resolution:
            self.width, self.height = self.resolution
        else:
            self.width, self.height = self.original_width, self.original_height
            
        # Configurer le FPS si spécifié
        if not self.fps:
            self.fps = self.original_fps
            
        # Initialiser le writer si un chemin de sortie est spécifié
        if self.output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                self.output_path, fourcc, self.fps, (self.width, self.height)
            )
            
        self.start_time = time.time()
        self.frame_count = 0
        
        return self
        
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Lit une frame de la vidéo et la prétraite.
        
        Returns:
            Tuple (success, frame)
        """
        if not self.cap or not self.cap.isOpened():
            return False, None
            
        success, frame = self.cap.read()
        if not success:
            return False, None
            
        # Redimensionner si nécessaire
        if self.resolution and (frame.shape[1] != self.width or frame.shape[0] != self.height):
            frame = cv2.resize(frame, (self.width, self.height))
            
        self.frame_count += 1
        
        # Calculer le FPS de traitement
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            self.processing_fps = self.frame_count / elapsed_time
            
        return True, frame
        
    def write(self, frame: np.ndarray):
        """
        Écrit une frame dans la vidéo de sortie si configurée.
        
        Args:
            frame: Image à écrire
        """
        if self.writer:
            self.writer.write(frame)
            
    def release(self):
        """Libère les ressources"""
        if self.cap:
            self.cap.release()
            
        if self.writer:
            self.writer.release()
            
        cv2.destroyAllWindows()
        
    def get_frame_info(self) -> Dict[str, Any]:
        """
        Retourne les informations sur le traitement des frames.
        
        Returns:
            Dictionnaire contenant les informations sur les frames
        """
        return {
            "frame_count": self.frame_count,
            "processing_fps": self.processing_fps,
            "original_resolution": (self.original_width, self.original_height),
            "current_resolution": (self.width, self.height),
            "original_fps": self.original_fps,
            "target_fps": self.fps
        }
        
    def process_video(self, process_frame_func: Callable[[np.ndarray, int], np.ndarray], 
                      display: bool = True, delay: int = 1):
        """
        Traite la vidéo frame par frame en utilisant la fonction fournie.
        
        Args:
            process_frame_func: Fonction qui prend une frame et son index et retourne la frame traitée
            display: Si True, affiche la vidéo pendant le traitement
            delay: Délai entre les frames en ms (1 par défaut, 0 pour le plus rapide possible)
        """
        try:
            while True:
                success, frame = self.read()
                if not success:
                    break
                    
                # Traiter la frame
                processed_frame = process_frame_func(frame, self.frame_count)
                
                # Afficher la frame si demandé
                if display:
                    cv2.imshow('Video Analysis', processed_frame)
                    
                # Écrire la frame si un writer est configuré
                self.write(processed_frame)
                
                # Attendre la touche 'q' pour quitter
                if cv2.waitKey(delay) & 0xFF == ord('q'):
                    break
                    
        finally:
            self.release()
