import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import os
import yaml
import sys
import time

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.video_processor import VideoProcessor
from detector.yolov8_detector import YOLOv8Detector
from utils.drawing import draw_bounding_boxes, draw_text, draw_alert

class FireSmokeDetector:
    """
    Classe pour détecter la fumée et le feu dans les flux vidéo.
    """
    def __init__(self, config_path: str):
        """
        Initialise le détecteur de fumée et de feu.
        
        Args:
            config_path: Chemin vers le fichier de configuration
        """
        # Charger la configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialiser le détecteur principal (pour fumée et feu si disponible)
        self.detector = YOLOv8Detector(
            model_path=self.config['detector']['model_path'],
            conf_threshold=self.config['detector']['conf_threshold'],
            iou_threshold=self.config['detector']['iou_threshold'],
            device=self.config['detector']['device']
        )
        
        # Configuration de la détection
        self.fire_smoke_config = self.config['fire_smoke_detection']
        
        # Classes à détecter (fumée, feu)
        self.target_classes = self.fire_smoke_config.get('target_classes', ['fire', 'smoke'])
        self.target_class_ids = None  # Sera initialisé lors de la première détection
        
        # État des alertes
        self.alert_active = False
        self.alert_start_time = None
        self.alert_history = []
        
        # Configuration des alertes
        self.alert_config = self.fire_smoke_config.get('alerts', {})
        self.min_detection_time = self.alert_config.get('min_detection_time', 3.0)  # secondes
        self.alert_cooldown = self.alert_config.get('cooldown', 10.0)  # secondes
        
        # Détection basée sur les couleurs (méthode de fallback)
        self.use_color_detection = self.fire_smoke_config.get('use_color_detection', True)
        self.color_thresholds = self.fire_smoke_config.get('color_thresholds', {
            'fire': {
                'lower_hsv': [0, 50, 50],
                'upper_hsv': [35, 255, 255]
            },
            'smoke': {
                'lower_gray': 100,
                'upper_gray': 200
            }
        })
        
        # Statistiques
        self.stats = {
            'fire_detections': 0,
            'smoke_detections': 0,
            'total_alerts': 0,
            'last_detection_time': None
        }
        
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        Traite une frame pour détecter la fumée et le feu.
        
        Args:
            frame: Image à traiter
            frame_idx: Index de la frame
            
        Returns:
            Image avec les visualisations et alertes
        """
        # Initialiser les IDs de classe cibles si nécessaire
        if self.target_class_ids is None and self.target_classes:
            self.target_class_ids = self.detector.get_class_ids(self.target_classes)
            
        # Détecter avec YOLOv8
        yolo_detections = []
        if self.target_class_ids:
            yolo_detections, frame_with_boxes = self.detector.detect_with_visualization(
                frame, classes=self.target_class_ids, draw=True
            )
        else:
            frame_with_boxes = frame.copy()
            
        # Détection basée sur les couleurs (méthode de fallback)
        color_detections = []
        if self.use_color_detection:
            color_detections = self._detect_by_color(frame)
            
        # Combiner les détections
        all_detections = yolo_detections + color_detections
        
        # Traiter les détections
        result = self._process_detections(frame_with_boxes, all_detections, frame_idx)
        
        # Dessiner les statistiques
        result = self._draw_statistics(result)
        
        return result
        
    def _detect_by_color(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Détecte la fumée et le feu basé sur l'analyse des couleurs.
        
        Args:
            frame: Image à analyser
            
        Returns:
            Liste des détections basées sur les couleurs
        """
        detections = []
        
        # Détection de feu basée sur les couleurs HSV
        if 'fire' in self.color_thresholds:
            fire_detections = self._detect_fire_by_color(frame)
            detections.extend(fire_detections)
            
        # Détection de fumée basée sur les niveaux de gris
        if 'smoke' in self.color_thresholds:
            smoke_detections = self._detect_smoke_by_color(frame)
            detections.extend(smoke_detections)
            
        return detections
        
    def _detect_fire_by_color(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Détecte le feu basé sur l'analyse des couleurs HSV.
        
        Args:
            frame: Image à analyser
            
        Returns:
            Liste des détections de feu
        """
        detections = []
        
        # Convertir en HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Seuils pour la détection de feu
        fire_config = self.color_thresholds['fire']
        lower_hsv = np.array(fire_config['lower_hsv'])
        upper_hsv = np.array(fire_config['upper_hsv'])
        
        # Créer le masque
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        
        # Appliquer des opérations morphologiques pour nettoyer le masque
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Trouver les contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrer les contours par taille
        min_area = 500  # Aire minimale pour considérer une détection
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # Calculer la boîte englobante
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculer la confiance basée sur l'aire et la forme
                confidence = min(0.8, area / 5000)  # Confiance basée sur l'aire
                
                detection = {
                    "bbox": [float(x), float(y), float(x + w), float(y + h)],
                    "confidence": confidence,
                    "class_id": 999,  # ID fictif pour le feu détecté par couleur
                    "class_name": "fire_color",
                    "detection_method": "color"
                }
                detections.append(detection)
                
        return detections
        
    def _detect_smoke_by_color(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Détecte la fumée basée sur l'analyse des niveaux de gris.
        
        Args:
            frame: Image à analyser
            
        Returns:
            Liste des détections de fumée
        """
        detections = []
        
        # Convertir en niveaux de gris
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Seuils pour la détection de fumée
        smoke_config = self.color_thresholds['smoke']
        lower_gray = smoke_config['lower_gray']
        upper_gray = smoke_config['upper_gray']
        
        # Créer le masque pour les zones grises (potentielle fumée)
        mask = cv2.inRange(gray, lower_gray, upper_gray)
        
        # Appliquer un flou pour détecter les zones diffuses (caractéristique de la fumée)
        blurred = cv2.GaussianBlur(mask, (15, 15), 0)
        
        # Trouver les contours
        contours, _ = cv2.findContours(blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filtrer les contours par taille et forme
        min_area = 1000  # Aire minimale pour considérer une détection de fumée
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                # Calculer la boîte englobante
                x, y, w, h = cv2.boundingRect(contour)
                
                # Vérifier le ratio largeur/hauteur (la fumée tend à être plus large que haute)
                aspect_ratio = w / h
                if aspect_ratio > 0.5:  # Filtrer les formes trop étroites
                    # Calculer la confiance
                    confidence = min(0.7, area / 10000)  # Confiance basée sur l'aire
                    
                    detection = {
                        "bbox": [float(x), float(y), float(x + w), float(y + h)],
                        "confidence": confidence,
                        "class_id": 998,  # ID fictif pour la fumée détectée par couleur
                        "class_name": "smoke_color",
                        "detection_method": "color"
                    }
                    detections.append(detection)
                    
        return detections
        
    def _process_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]], 
                           frame_idx: int) -> np.ndarray:
        """
        Traite les détections et gère les alertes.
        
        Args:
            frame: Image à traiter
            detections: Liste des détections
            frame_idx: Index de la frame
            
        Returns:
            Image avec les visualisations et alertes
        """
        result = frame.copy()
        current_time = time.time()
        
        # Séparer les détections par type
        fire_detections = [d for d in detections if 'fire' in d['class_name'].lower()]
        smoke_detections = [d for d in detections if 'smoke' in d['class_name'].lower()]
        
        # Mettre à jour les statistiques
        if fire_detections:
            self.stats['fire_detections'] += len(fire_detections)
            self.stats['last_detection_time'] = current_time
            
        if smoke_detections:
            self.stats['smoke_detections'] += len(smoke_detections)
            self.stats['last_detection_time'] = current_time
            
        # Dessiner les détections avec des couleurs spécifiques
        for detection in fire_detections:
            result = self._draw_detection(result, detection, (0, 0, 255))  # Rouge pour le feu
            
        for detection in smoke_detections:
            result = self._draw_detection(result, detection, (128, 128, 128))  # Gris pour la fumée
            
        # Gérer les alertes
        if fire_detections or smoke_detections:
            if not self.alert_active:
                self.alert_start_time = current_time
                self.alert_active = True
                
            # Vérifier si l'alerte doit être déclenchée
            alert_duration = current_time - self.alert_start_time
            if alert_duration >= self.min_detection_time:
                # Déclencher l'alerte
                alert_type = "FEU" if fire_detections else "FUMÉE"
                if fire_detections and smoke_detections:
                    alert_type = "FEU ET FUMÉE"
                    
                alert_message = f"ALERTE {alert_type} DÉTECTÉ!"
                result = draw_alert(result, alert_message, blink=True, frame_count=frame_idx)
                
                # Ajouter à l'historique
                if len(self.alert_history) == 0 or current_time - self.alert_history[-1]['timestamp'] > self.alert_cooldown:
                    self.alert_history.append({
                        'timestamp': current_time,
                        'type': alert_type,
                        'fire_count': len(fire_detections),
                        'smoke_count': len(smoke_detections)
                    })
                    self.stats['total_alerts'] += 1
        else:
            # Réinitialiser l'alerte si aucune détection
            self.alert_active = False
            self.alert_start_time = None
            
        return result
        
    def _draw_detection(self, frame: np.ndarray, detection: Dict[str, Any], 
                       color: Tuple[int, int, int]) -> np.ndarray:
        """
        Dessine une détection sur la frame.
        
        Args:
            frame: Image sur laquelle dessiner
            detection: Informations de la détection
            color: Couleur de la boîte englobante
            
        Returns:
            Image avec la détection dessinée
        """
        result = frame.copy()
        
        x1, y1, x2, y2 = map(int, detection["bbox"])
        conf = detection["confidence"]
        class_name = detection["class_name"]
        method = detection.get("detection_method", "yolo")
        
        # Créer le label
        if method == "color":
            label = f"{class_name} (couleur) {conf:.2f}"
        else:
            label = f"{class_name} {conf:.2f}"
            
        # Dessiner la boîte englobante
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 3)
        
        # Dessiner l'étiquette
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(result, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
        cv2.putText(result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return result
        
    def _draw_statistics(self, frame: np.ndarray) -> np.ndarray:
        """
        Dessine les statistiques sur la frame.
        
        Args:
            frame: Image sur laquelle dessiner
            
        Returns:
            Image avec les statistiques
        """
        result = frame.copy()
        
        # Position de départ pour les statistiques
        x, y = 10, 30
        line_height = 25
        
        # Statistiques de détection
        result = draw_text(result, f"Détections feu: {self.stats['fire_detections']}", (x, y), 
                         font_scale=0.6, color=(255, 255, 255))
        y += line_height
        
        result = draw_text(result, f"Détections fumée: {self.stats['smoke_detections']}", (x, y), 
                         font_scale=0.6, color=(255, 255, 255))
        y += line_height
        
        result = draw_text(result, f"Total alertes: {self.stats['total_alerts']}", (x, y), 
                         font_scale=0.6, color=(255, 255, 255))
        y += line_height
        
        # État de l'alerte
        if self.alert_active:
            result = draw_text(result, "ÉTAT: ALERTE ACTIVE", (x, y), 
                             font_scale=0.6, color=(0, 0, 255))
        else:
            result = draw_text(result, "ÉTAT: SURVEILLANCE", (x, y), 
                             font_scale=0.6, color=(0, 255, 0))
            
        return result
        
def main():
    """Fonction principale pour exécuter le détecteur de fumée et de feu."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Détecteur de fumée et de feu")
    parser.add_argument('--config', type=str, default='configs/fire_smoke_detection.yaml',
                        help='Chemin vers le fichier de configuration')
    parser.add_argument('--source', type=str, default='0',
                        help='Source vidéo (0 pour webcam, chemin pour fichier)')
    parser.add_argument('--output', type=str, default=None,
                        help='Chemin pour sauvegarder la vidéo traitée')
    args = parser.parse_args()
    
    # Initialiser le détecteur de fumée et de feu
    detector = FireSmokeDetector(args.config)
    
    # Initialiser le processeur vidéo
    with VideoProcessor(args.source, args.output) as video_proc:
        # Traiter la vidéo
        video_proc.process_video(detector.process_frame)
        
if __name__ == "__main__":
    main()
