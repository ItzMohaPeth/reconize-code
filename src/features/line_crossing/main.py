import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import os
import yaml
import sys

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.video_processor import VideoProcessor
from detector.yolov8_detector import YOLOv8Detector
from core.object_tracker import ObjectTracker
from utils.drawing import draw_bounding_boxes, draw_trajectories, draw_line, draw_counter, draw_text

class LineCrossingCounter:
    """
    Classe pour compter les personnes traversant une ligne définie.
    """
    def __init__(self, config_path: str):
        """
        Initialise le compteur de traversée de ligne.
        
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
        
        # Configurer la ligne de comptage
        self.line = self.config['line_crossing']['line']
        self.line_start = tuple(self.line['start'])
        self.line_end = tuple(self.line['end'])
        
        # Calculer le vecteur normal à la ligne (pour déterminer la direction)
        line_vector = np.array([self.line_end[0] - self.line_start[0], self.line_end[1] - self.line_start[1]])
        self.normal_vector = np.array([-line_vector[1], line_vector[0]])
        self.normal_vector = self.normal_vector / np.linalg.norm(self.normal_vector)
        
        # Initialiser les compteurs
        self.counts = {
            'up': 0,
            'down': 0,
            'total': 0
        }
        
        # Garder une trace des objets qui ont déjà traversé la ligne
        self.crossed_objects = set()
        
        # Classes à détecter (par défaut: personne)
        self.target_classes = self.config['detector'].get('target_classes', ['person'])
        self.target_class_ids = None  # Sera initialisé lors de la première détection
        
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        Traite une frame pour détecter et compter les traversées de ligne.
        
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
        
        # Dessiner les trajectoires
        result = draw_trajectories(frame_with_boxes, self.tracker)
        
        # Dessiner la ligne de comptage
        result = draw_line(result, self.line_start, self.line_end, color=(0, 255, 255), thickness=2)
        
        # Vérifier les traversées de ligne
        self._check_line_crossings(tracked_objects)
        
        # Dessiner les compteurs
        result = draw_counter(result, "Montant", self.counts['up'], (10, 30))
        result = draw_counter(result, "Descendant", self.counts['down'], (10, 70))
        result = draw_counter(result, "Total", self.counts['total'], (10, 110))
        
        # Afficher les informations de traitement
        fps_text = f"FPS: {frame_idx / max(1, self.tracker.current_frame / 30):.1f}"
        result = draw_text(result, fps_text, (result.shape[1] - 150, 30))
        
        return result
        
    def _check_line_crossings(self, tracked_objects: List[Dict[str, Any]]):
        """
        Vérifie si des objets ont traversé la ligne et met à jour les compteurs.
        
        Args:
            tracked_objects: Liste des objets suivis
        """
        for obj in tracked_objects:
            object_id = obj['track_id']
            
            # Ignorer les objets qui ont déjà traversé la ligne
            if object_id in self.crossed_objects:
                continue
                
            # Récupérer la trajectoire de l'objet
            trajectory = self.tracker.get_object_trajectory(object_id)
            
            # Vérifier s'il y a au moins deux points dans la trajectoire
            if len(trajectory) < 2:
                continue
                
            # Vérifier si l'objet a traversé la ligne entre les deux derniers points
            prev_point = trajectory[-2]
            curr_point = trajectory[-1]
            
            if self._line_crossing_check(prev_point, curr_point):
                # Déterminer la direction de la traversée
                direction = self._determine_crossing_direction(prev_point, curr_point)
                
                # Mettre à jour les compteurs
                if direction > 0:
                    self.counts['up'] += 1
                else:
                    self.counts['down'] += 1
                    
                self.counts['total'] += 1
                
                # Marquer l'objet comme ayant traversé la ligne
                self.crossed_objects.add(object_id)
                
    def _line_crossing_check(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> bool:
        """
        Vérifie si le segment [p1, p2] traverse la ligne de comptage.
        
        Args:
            p1: Premier point du segment
            p2: Deuxième point du segment
            
        Returns:
            True si le segment traverse la ligne, False sinon
        """
        # Fonction pour vérifier si un point est à gauche d'une ligne orientée
        def is_left(line_start, line_end, point):
            return ((line_end[0] - line_start[0]) * (point[1] - line_start[1]) - 
                    (line_end[1] - line_start[1]) * (point[0] - line_start[0]))
                    
        # Vérifier si les points sont de part et d'autre de la ligne
        return (is_left(self.line_start, self.line_end, p1) * 
                is_left(self.line_start, self.line_end, p2)) < 0
                
    def _determine_crossing_direction(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> int:
        """
        Détermine la direction de traversée par rapport au vecteur normal à la ligne.
        
        Args:
            p1: Point avant la traversée
            p2: Point après la traversée
            
        Returns:
            1 pour une traversée dans le sens du vecteur normal, -1 pour le sens opposé
        """
        # Calculer le vecteur de déplacement
        movement_vector = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        
        # Normaliser le vecteur
        if np.linalg.norm(movement_vector) > 0:
            movement_vector = movement_vector / np.linalg.norm(movement_vector)
            
        # Calculer le produit scalaire avec le vecteur normal
        dot_product = np.dot(movement_vector, self.normal_vector)
        
        # Retourner la direction
        return 1 if dot_product > 0 else -1
        
def main():
    """Fonction principale pour exécuter le compteur de traversée de ligne."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Compteur de traversée de ligne")
    parser.add_argument('--config', type=str, default='configs/line_crossing.yaml',
                        help='Chemin vers le fichier de configuration')
    parser.add_argument('--source', type=str, default='0',
                        help='Source vidéo (0 pour webcam, chemin pour fichier)')
    parser.add_argument('--output', type=str, default=None,
                        help='Chemin pour sauvegarder la vidéo traitée')
    args = parser.parse_args()
    
    # Initialiser le compteur de traversée de ligne
    counter = LineCrossingCounter(args.config)
    
    # Initialiser le processeur vidéo
    with VideoProcessor(args.source, args.output) as video_proc:
        # Traiter la vidéo
        video_proc.process_video(counter.process_frame)
        
if __name__ == "__main__":
    main()
