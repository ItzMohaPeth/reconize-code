import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
import os
import yaml
import sys

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.video_processor import VideoProcessor
from detector.yolov8_detector import YOLOv8Detector
from core.object_tracker import ObjectTracker
from utils.drawing import draw_bounding_boxes, draw_trajectories, draw_polygon, draw_counter, draw_text

class ZoneCounter:
    """
    Classe pour compter les personnes se déplaçant entre des zones prédéfinies.
    """
    def __init__(self, config_path: str):
        """
        Initialise le compteur de zones.
        
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
        
        # Configurer les zones
        self.zones = self.config['zone_counting']['zones']
        self.zone_colors = {
            zone_id: tuple(self.zones[zone_id].get('color', (0, 255, 0)))
            for zone_id in self.zones
        }
        
        # Initialiser les compteurs de transition entre zones
        self.transitions = {}
        for source_zone in self.zones:
            for target_zone in self.zones:
                if source_zone != target_zone:
                    self.transitions[(source_zone, target_zone)] = 0
                    
        # Garder une trace de la dernière zone de chaque objet
        self.object_zones = {}  # {object_id: zone_id}
        
        # Classes à détecter (par défaut: personne)
        self.target_classes = self.config['detector'].get('target_classes', ['person'])
        self.target_class_ids = None  # Sera initialisé lors de la première détection
        
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        Traite une frame pour détecter et compter les déplacements entre zones.
        
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
        
        # Dessiner les zones
        result = frame_with_boxes.copy()
        for zone_id, zone_info in self.zones.items():
            points = zone_info['points']
            color = tuple(zone_info.get('color', (0, 255, 0)))
            result = draw_polygon(result, points, color=color, thickness=2, fill=True)
            
        # Vérifier les zones des objets et les transitions
        self._check_zone_transitions(tracked_objects)
        
        # Dessiner les compteurs de transition
        y_offset = 30
        for (source, target), count in self.transitions.items():
            if count > 0:  # N'afficher que les transitions qui ont eu lieu
                text = f"{source} → {target}: {count}"
                result = draw_text(result, text, (10, y_offset))
                y_offset += 30
                
        # Afficher les informations de traitement
        fps_text = f"FPS: {frame_idx / max(1, self.tracker.current_frame / 30):.1f}"
        result = draw_text(result, fps_text, (result.shape[1] - 150, 30))
        
        return result
        
    def _check_zone_transitions(self, tracked_objects: List[Dict[str, Any]]):
        """
        Vérifie si des objets ont changé de zone et met à jour les compteurs.
        
        Args:
            tracked_objects: Liste des objets suivis
        """
        for obj in tracked_objects:
            object_id = obj['track_id']
            centroid = self._get_centroid(obj['bbox'])
            
            # Déterminer la zone actuelle de l'objet
            current_zone = self._get_zone(centroid)
            
            # Si l'objet n'était pas encore suivi ou n'était dans aucune zone
            if object_id not in self.object_zones:
                if current_zone:
                    self.object_zones[object_id] = current_zone
                continue
                
            # Si l'objet a changé de zone
            previous_zone = self.object_zones[object_id]
            if current_zone and current_zone != previous_zone:
                # Mettre à jour le compteur de transition
                self.transitions[(previous_zone, current_zone)] += 1
                
                # Mettre à jour la zone actuelle de l'objet
                self.object_zones[object_id] = current_zone
                
    def _get_zone(self, point: Tuple[float, float]) -> Optional[str]:
        """
        Détermine dans quelle zone se trouve un point.
        
        Args:
            point: Coordonnées (x, y) du point
            
        Returns:
            ID de la zone ou None si le point n'est dans aucune zone
        """
        for zone_id, zone_info in self.zones.items():
            if self._point_in_polygon(point, zone_info['points']):
                return zone_id
        return None
        
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
    """Fonction principale pour exécuter le compteur de zones."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compteur de zones")
    parser.add_argument('--config', type=str, default='configs/zone_counting.yaml',
                        help='Chemin vers le fichier de configuration')
    parser.add_argument('--source', type=str, default='0',
                        help='Source vidéo (0 pour webcam, chemin pour fichier)')
    parser.add_argument('--output', type=str, default=None,
                        help='Chemin pour sauvegarder la vidéo traitée')
    args = parser.parse_args()
    
    # Initialiser le compteur de zones
    counter = ZoneCounter(args.config)
    
    # Initialiser le processeur vidéo
    with VideoProcessor(args.source, args.output) as video_proc:
        # Traiter la vidéo
        video_proc.process_video(counter.process_frame)
        
if __name__ == "__main__":
    main()
