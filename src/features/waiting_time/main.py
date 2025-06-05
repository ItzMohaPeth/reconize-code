import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
import os
import yaml
import sys
import time

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.video_processor import VideoProcessor
from detector.yolov8_detector import YOLOv8Detector
from core.object_tracker import ObjectTracker
from utils.drawing import draw_bounding_boxes, draw_trajectories, draw_polygon, draw_text

class WaitingTimeAnalyzer:
    """
    Classe pour analyser le temps d'attente des personnes dans une file.
    """
    def __init__(self, config_path: str):
        """
        Initialise l'analyseur de temps d'attente.
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        # Charger la configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialiser le détecteur
        self.detector = YOLOv8Detector(
            model_path=self.config['detector']['model_path'],
            conf_threshold=self.config['detector']['conf_threshold'],
            iou_threshold=self.config['detector']['iou_threshold'],
            device=self.config['detector']['device']
        )
        
        # Initialiser le tracker
        self.tracker = ObjectTracker(
            max_disappeared=self.config['tracker']['max_disappeared'],
            min_iou_threshold=self.config['tracker']['min_iou_threshold']
        )
        
        # Configurer la zone de file d'attente
        self.queue_zone = self.config['waiting_time']['queue_zone']
        self.queue_color = tuple(self.queue_zone.get('color', (0, 255, 255)))
        
        # Garder une trace du temps d'attente de chaque objet
        self.waiting_times = {}  # {object_id: {'start_time': timestamp, 'total_time': seconds}}
        
        # Statistiques sur les temps d'attente
        self.stats = {
            'current_avg_time': 0,
            'max_time': 0,
            'total_people': 0,
            'served_people': 0
        }
        
        # Temps réel ou temps de la vidéo
        self.use_real_time = self.config['waiting_time'].get('use_real_time', True)
        self.video_fps = 30  # Sera mis à jour avec les vraies valeurs
        
        # Classes à détecter (par défaut: personne)
        self.target_classes = self.config['detector'].get('target_classes', ['person'])
        self.target_class_ids = None  # Sera initialisé lors de la première détection
        
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        Traite une frame pour analyser les temps d'attente.
        
        Args:
            frame: Image à traiter
            frame_idx: Index de la frame
            
        Returns:
            Image avec les visualisations
        """
        # Initialiser les IDs de classe cibles si nécessaire
        if self.target_class_ids is None and self.target_classes:
            self.target_class_ids = self.detector.get_class_ids(self.target_classes)
            
        # Détecter les objets
        detections, frame_with_boxes = self.detector.detect_with_visualization(
            frame, classes=self.target_class_ids, draw=True
        )
        
        # Mettre à jour le tracker
        tracked_objects = self.tracker.update(detections, frame_idx)
        
        # Dessiner la zone de file d'attente
        result = frame_with_boxes.copy()
        result = draw_polygon(result, self.queue_zone['points'], 
                             color=self.queue_color, thickness=2, fill=True)
        
        # Analyser les temps d'attente
        self._analyze_waiting_times(tracked_objects, frame_idx)
        
        # Dessiner les informations de temps d'attente sur chaque personne
        result = self._draw_waiting_info(result, tracked_objects)
        
        # Dessiner les statistiques globales
        result = self._draw_statistics(result)
        
        return result
        
    def _analyze_waiting_times(self, tracked_objects: List[Dict[str, Any]], frame_idx: int):
        """
        Analyse les temps d'attente des objets dans la zone de file.
        
        Args:
            tracked_objects: Liste des objets suivis
            frame_idx: Index de la frame actuelle
        """
        current_time = self._get_current_time(frame_idx)
        objects_in_queue = set()
        
        for obj in tracked_objects:
            object_id = obj['track_id']
            centroid = self._get_centroid(obj['bbox'])
            
            # Vérifier si l'objet est dans la zone de file
            in_queue = self._point_in_polygon(centroid, self.queue_zone['points'])
            
            if in_queue:
                objects_in_queue.add(object_id)
                
                # Si l'objet vient d'entrer dans la file
                if object_id not in self.waiting_times:
                    self.waiting_times[object_id] = {
                        'start_time': current_time,
                        'total_time': 0
                    }
                    self.stats['total_people'] += 1
                    
                # Mettre à jour le temps d'attente
                self.waiting_times[object_id]['total_time'] = current_time - self.waiting_times[object_id]['start_time']
                
        # Gérer les objets qui ont quitté la file
        for object_id in list(self.waiting_times.keys()):
            if object_id not in objects_in_queue and object_id in self.waiting_times:
                # L'objet a quitté la file
                final_time = self.waiting_times[object_id]['total_time']
                
                # Mettre à jour les statistiques
                self.stats['served_people'] += 1
                if final_time > self.stats['max_time']:
                    self.stats['max_time'] = final_time
                    
                # Supprimer de la liste des objets en attente
                del self.waiting_times[object_id]
                
        # Calculer le temps d'attente moyen actuel
        if self.waiting_times:
            total_current_time = sum(wt['total_time'] for wt in self.waiting_times.values())
            self.stats['current_avg_time'] = total_current_time / len(self.waiting_times)
        else:
            self.stats['current_avg_time'] = 0
            
    def _draw_waiting_info(self, frame: np.ndarray, tracked_objects: List[Dict[str, Any]]) -> np.ndarray:
        """
        Dessine les informations de temps d'attente sur chaque personne.
        
        Args:
            frame: Image sur laquelle dessiner
            tracked_objects: Liste des objets suivis
            
        Returns:
            Image avec les informations de temps d'attente
        """
        result = frame.copy()
        
        for obj in tracked_objects:
            object_id = obj['track_id']
            
            if object_id in self.waiting_times:
                # Calculer la position pour afficher le temps
                x1, y1, x2, y2 = map(int, obj['bbox'])
                
                # Temps d'attente en secondes
                waiting_time = self.waiting_times[object_id]['total_time']
                
                # Formater le temps d'attente
                if waiting_time < 60:
                    time_text = f"{waiting_time:.1f}s"
                else:
                    minutes = int(waiting_time // 60)
                    seconds = int(waiting_time % 60)
                    time_text = f"{minutes}m{seconds:02d}s"
                    
                # Dessiner le temps d'attente
                text_position = (x1, y1 - 10)
                result = draw_text(result, time_text, text_position, 
                                 font_scale=0.6, color=(255, 255, 0))
                
        return result
        
    def _draw_statistics(self, frame: np.ndarray) -> np.ndarray:
        """
        Dessine les statistiques globales sur la frame.
        
        Args:
            frame: Image sur laquelle dessiner
            
        Returns:
            Image avec les statistiques
        """
        result = frame.copy()
        
        # Position de départ pour les statistiques
        x, y = 10, 30
        line_height = 30
        
        # Nombre de personnes actuellement en attente
        current_waiting = len(self.waiting_times)
        result = draw_text(result, f"En attente: {current_waiting}", (x, y))
        y += line_height
        
        # Temps d'attente moyen actuel
        if self.stats['current_avg_time'] > 0:
            avg_time = self.stats['current_avg_time']
            if avg_time < 60:
                avg_text = f"Temps moyen: {avg_time:.1f}s"
            else:
                minutes = int(avg_time // 60)
                seconds = int(avg_time % 60)
                avg_text = f"Temps moyen: {minutes}m{seconds:02d}s"
            result = draw_text(result, avg_text, (x, y))
            y += line_height
            
        # Temps d'attente maximum
        if self.stats['max_time'] > 0:
            max_time = self.stats['max_time']
            if max_time < 60:
                max_text = f"Temps max: {max_time:.1f}s"
            else:
                minutes = int(max_time // 60)
                seconds = int(max_time % 60)
                max_text = f"Temps max: {minutes}m{seconds:02d}s"
            result = draw_text(result, max_text, (x, y))
            y += line_height
            
        # Nombre total de personnes servies
        result = draw_text(result, f"Personnes servies: {self.stats['served_people']}", (x, y))
        
        return result
        
    def _get_current_time(self, frame_idx: int) -> float:
        """
        Obtient le temps actuel (réel ou basé sur la vidéo).
        
        Args:
            frame_idx: Index de la frame actuelle
            
        Returns:
            Temps en secondes
        """
        if self.use_real_time:
            return time.time()
        else:
            # Temps basé sur l'index de la frame et le FPS
            return frame_idx / self.video_fps
            
    def _point_in_polygon(self, point: Tuple[float, float], polygon: List[Tuple[int, int]]) -> bool:
        """
        Vérifie si un point est à l'intérieur d'un polygone.
        
        Args:
            point: Coordonnées (x, y) du point
            polygon: Liste des sommets du polygone [(x1, y1), (x2, y2), ...]
            
        Returns:
            True si le point est dans le polygone, False sinon
        """
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside
        
    def _get_centroid(self, bbox: List[float]) -> Tuple[float, float]:
        """
        Calcule le centroïde d'une boîte englobante.
        
        Args:
            bbox: [x1, y1, x2, y2] de la boîte
            
        Returns:
            Tuple (x, y) du centroïde
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
        
def main():
    """Fonction principale pour exécuter l'analyseur de temps d'attente."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyseur de temps d'attente")
    parser.add_argument('--config', type=str, default='configs/waiting_time.yaml',
                        help='Chemin vers le fichier de configuration')
    parser.add_argument('--source', type=str, default='0',
                        help='Source vidéo (0 pour webcam, chemin pour fichier)')
    parser.add_argument('--output', type=str, default=None,
                        help='Chemin pour sauvegarder la vidéo traitée')
    args = parser.parse_args()
    
    # Initialiser l'analyseur de temps d'attente
    analyzer = WaitingTimeAnalyzer(args.config)
    
    # Initialiser le processeur vidéo
    with VideoProcessor(args.source, args.output) as video_proc:
        # Traiter la vidéo
        video_proc.process_video(analyzer.process_frame)
        
if __name__ == "__main__":
    main()
