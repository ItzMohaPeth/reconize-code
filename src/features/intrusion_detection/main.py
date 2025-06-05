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
from utils.drawing import draw_bounding_boxes, draw_trajectories, draw_polygon, draw_text, draw_alert

class IntrusionDetector:
    """
    Classe pour détecter les intrusions dans des zones interdites.
    """
    def __init__(self, config_path: str):
        """
        Initialise le détecteur d'intrusion.
        
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
        
        # Configurer les zones interdites
        self.restricted_zones = self.config['intrusion_detection']['restricted_zones']
        
        # État des alertes
        self.active_alerts = set()  # IDs des objets en intrusion
        self.alert_history = []  # Historique des alertes
        
        # Configuration des alertes
        self.alert_config = self.config['intrusion_detection'].get('alerts', {})
        self.min_alert_duration = self.alert_config.get('min_duration', 2.0)  # secondes
        self.alert_cooldown = self.alert_config.get('cooldown', 5.0)  # secondes
        
        # Temps des alertes par objet
        self.alert_times = {}  # {object_id: {'start_time': timestamp, 'zone': zone_id}}
        
        # Classes à détecter (par défaut: personne)
        self.target_classes = self.config['detector'].get('target_classes', ['person'])
        self.target_class_ids = None  # Sera initialisé lors de la première détection
        
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        Traite une frame pour détecter les intrusions.
        
        Args:
            frame: Image à traiter
            frame_idx: Index de la frame
            
        Returns:
            Image avec les visualisations et alertes
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
        
        # Dessiner les zones interdites
        result = frame_with_boxes.copy()
        for zone_id, zone_info in self.restricted_zones.items():
            points = zone_info['points']
            color = tuple(zone_info.get('color', (0, 0, 255)))  # Rouge par défaut
            result = draw_polygon(result, points, color=color, thickness=2, fill=True)
            
            # Ajouter le nom de la zone
            if 'name' in zone_info:
                # Calculer le centre de la zone pour placer le texte
                center_x = sum(p[0] for p in points) // len(points)
                center_y = sum(p[1] for p in points) // len(points)
                result = draw_text(result, zone_info['name'], (center_x, center_y), 
                                 font_scale=0.7, color=(255, 255, 255))
                
        # Détecter les intrusions
        intrusions = self._detect_intrusions(tracked_objects, frame_idx)
        
        # Dessiner les alertes d'intrusion
        result = self._draw_intrusion_alerts(result, intrusions, frame_idx)
        
        # Dessiner les statistiques
        result = self._draw_statistics(result)
        
        return result
        
    def _detect_intrusions(self, tracked_objects: List[Dict[str, Any]], frame_idx: int) -> List[Dict[str, Any]]:
        """
        Détecte les intrusions dans les zones interdites.
        
        Args:
            tracked_objects: Liste des objets suivis
            frame_idx: Index de la frame actuelle
            
        Returns:
            Liste des intrusions détectées
        """
        current_time = time.time()
        intrusions = []
        current_intruders = set()
        
        for obj in tracked_objects:
            object_id = obj['track_id']
            centroid = self._get_centroid(obj['bbox'])
            
            # Vérifier si l'objet est dans une zone interdite
            for zone_id, zone_info in self.restricted_zones.items():
                if self._point_in_polygon(centroid, zone_info['points']):
                    current_intruders.add(object_id)
                    
                    # Si c'est une nouvelle intrusion
                    if object_id not in self.alert_times:
                        self.alert_times[object_id] = {
                            'start_time': current_time,
                            'zone': zone_id
                        }
                        
                    # Vérifier si l'intrusion dure assez longtemps pour déclencher une alerte
                    duration = current_time - self.alert_times[object_id]['start_time']
                    if duration >= self.min_alert_duration:
                        intrusion_info = {
                            'object_id': object_id,
                            'zone_id': zone_id,
                            'zone_name': zone_info.get('name', f'Zone {zone_id}'),
                            'duration': duration,
                            'bbox': obj['bbox'],
                            'centroid': centroid
                        }
                        intrusions.append(intrusion_info)
                        
                        # Ajouter à l'historique si ce n'est pas déjà fait récemment
                        if object_id not in self.active_alerts:
                            self.alert_history.append({
                                'timestamp': current_time,
                                'object_id': object_id,
                                'zone_id': zone_id,
                                'zone_name': zone_info.get('name', f'Zone {zone_id}')
                            })
                            
                        self.active_alerts.add(object_id)
                    break  # Un objet ne peut être que dans une zone à la fois
                    
        # Nettoyer les alertes pour les objets qui ne sont plus en intrusion
        for object_id in list(self.alert_times.keys()):
            if object_id not in current_intruders:
                del self.alert_times[object_id]
                self.active_alerts.discard(object_id)
                
        return intrusions
        
    def _draw_intrusion_alerts(self, frame: np.ndarray, intrusions: List[Dict[str, Any]], 
                              frame_idx: int) -> np.ndarray:
        """
        Dessine les alertes d'intrusion sur la frame.
        
        Args:
            frame: Image sur laquelle dessiner
            intrusions: Liste des intrusions détectées
            frame_idx: Index de la frame actuelle
            
        Returns:
            Image avec les alertes d'intrusion
        """
        result = frame.copy()
        
        # Dessiner les boîtes d'alerte pour chaque intrus
        for intrusion in intrusions:
            x1, y1, x2, y2 = map(int, intrusion['bbox'])
            
            # Dessiner une boîte rouge clignotante autour de l'intrus
            if (frame_idx // 10) % 2 == 0:  # Clignote toutes les 10 frames
                cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 4)
                
            # Afficher les informations de l'intrusion
            alert_text = f"INTRUSION! {intrusion['zone_name']}"
            duration_text = f"Durée: {intrusion['duration']:.1f}s"
            
            result = draw_text(result, alert_text, (x1, y1 - 40), 
                             font_scale=0.7, color=(0, 0, 255))
            result = draw_text(result, duration_text, (x1, y1 - 20), 
                             font_scale=0.5, color=(0, 0, 255))
                             
        # Afficher une alerte générale si il y a des intrusions
        if intrusions:
            alert_message = f"ALERTE: {len(intrusions)} INTRUSION(S) DÉTECTÉE(S)"
            result = draw_alert(result, alert_message, blink=True, frame_count=frame_idx)
            
        return result
        
    def _draw_statistics(self, frame: np.ndarray) -> np.ndarray:
        """
        Dessine les statistiques sur les intrusions.
        
        Args:
            frame: Image sur laquelle dessiner
            
        Returns:
            Image avec les statistiques
        """
        result = frame.copy()
        
        # Position de départ pour les statistiques
        x, y = 10, frame.shape[0] - 100
        line_height = 20
        
        # Nombre d'intrusions actives
        active_count = len(self.active_alerts)
        result = draw_text(result, f"Intrusions actives: {active_count}", (x, y), 
                         font_scale=0.6, color=(255, 255, 255))
        y += line_height
        
        # Nombre total d'intrusions détectées
        total_count = len(self.alert_history)
        result = draw_text(result, f"Total intrusions: {total_count}", (x, y), 
                         font_scale=0.6, color=(255, 255, 255))
        y += line_height
        
        # Dernière intrusion
        if self.alert_history:
            last_alert = self.alert_history[-1]
            last_text = f"Dernière: {last_alert['zone_name']}"
            result = draw_text(result, last_text, (x, y), 
                             font_scale=0.6, color=(255, 255, 255))
            
        return result
        
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
    """Fonction principale pour exécuter le détecteur d'intrusion."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Détecteur d'intrusion")
    parser.add_argument('--config', type=str, default='configs/intrusion_detection.yaml',
                        help='Chemin vers le fichier de configuration')
    parser.add_argument('--source', type=str, default='0',
                        help='Source vidéo (0 pour webcam, chemin pour fichier)')
    parser.add_argument('--output', type=str, default=None,
                        help='Chemin pour sauvegarder la vidéo traitée')
    args = parser.parse_args()
    
    # Initialiser le détecteur d'intrusion
    detector = IntrusionDetector(args.config)
    
    # Initialiser le processeur vidéo
    with VideoProcessor(args.source, args.output) as video_proc:
        # Traiter la vidéo
        video_proc.process_video(detector.process_frame)
        
if __name__ == "__main__":
    main()
