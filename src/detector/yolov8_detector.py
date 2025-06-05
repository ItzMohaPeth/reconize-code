from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import cv2
import os

class YOLOv8Detector:
    """
    Classe pour encapsuler la détection d'objets avec YOLOv8.
    """
    def __init__(self, model_path: str, conf_threshold: float = 0.25, 
                 iou_threshold: float = 0.45, device: str = 'auto'):
        """
        Initialise le détecteur YOLOv8.
        
        Args:
            model_path: Chemin vers le fichier de poids du modèle YOLOv8
            conf_threshold: Seuil de confiance pour la détection
            iou_threshold: Seuil IoU pour le NMS
            device: Appareil à utiliser ('cpu', 'cuda:0', 'auto')
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Le fichier de modèle n'existe pas: {model_path}")
            
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.class_names = self.model.names
        
    def detect(self, frame: np.ndarray, classes: Optional[List[int]] = None) -> List[Dict[str, Any]]:
        """
        Détecte les objets dans une frame.
        
        Args:
            frame: Image à analyser
            classes: Liste des IDs de classes à détecter (None pour toutes les classes)
            
        Returns:
            Liste de dictionnaires contenant les détections
        """
        # Effectuer la prédiction
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            classes=classes,
            device=self.device,
            verbose=False
        )
        
        # Extraire les résultats
        detections = []
        
        if results and len(results) > 0:
            result = results[0]  # Premier résultat (une seule image)
            
            # Convertir les tensors en numpy arrays si nécessaire
            boxes = result.boxes.xyxy.cpu().numpy() if hasattr(result.boxes, 'xyxy') else []
            confs = result.boxes.conf.cpu().numpy() if hasattr(result.boxes, 'conf') else []
            cls_ids = result.boxes.cls.cpu().numpy().astype(int) if hasattr(result.boxes, 'cls') else []
            
            # Créer la liste des détections
            for i, (box, conf, cls_id) in enumerate(zip(boxes, confs, cls_ids)):
                x1, y1, x2, y2 = box
                detection = {
                    "id": i,
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(conf),
                    "class_id": int(cls_id),
                    "class_name": self.class_names[cls_id] if cls_id in self.class_names else f"class_{cls_id}"
                }
                detections.append(detection)
                
        return detections
        
    def detect_with_visualization(self, frame: np.ndarray, classes: Optional[List[int]] = None, 
                                  draw: bool = True) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Détecte les objets et visualise les résultats sur la frame.
        
        Args:
            frame: Image à analyser
            classes: Liste des IDs de classes à détecter (None pour toutes les classes)
            draw: Si True, dessine les boîtes englobantes sur la frame
            
        Returns:
            Tuple (détections, frame avec visualisation)
        """
        detections = self.detect(frame, classes)
        
        if draw:
            frame_with_boxes = frame.copy()
            for det in detections:
                x1, y1, x2, y2 = map(int, det["bbox"])
                cls_id = det["class_id"]
                conf = det["confidence"]
                label = f"{det['class_name']} {conf:.2f}"
                
                # Générer une couleur basée sur l'ID de classe
                color = self._get_color_by_class_id(cls_id)
                
                # Dessiner la boîte englobante
                cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, 2)
                
                # Dessiner l'étiquette
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                cv2.rectangle(frame_with_boxes, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), color, -1)
                cv2.putText(frame_with_boxes, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
            return detections, frame_with_boxes
        
        return detections, frame
        
    def _get_color_by_class_id(self, class_id: int) -> Tuple[int, int, int]:
        """
        Génère une couleur unique basée sur l'ID de classe.
        
        Args:
            class_id: ID de la classe
            
        Returns:
            Tuple (B, G, R) représentant la couleur
        """
        # Liste de couleurs prédéfinies pour les premières classes
        colors = [
            (0, 255, 0),    # Vert
            (0, 0, 255),    # Rouge
            (255, 0, 0),    # Bleu
            (255, 255, 0),  # Cyan
            (0, 255, 255),  # Jaune
            (255, 0, 255),  # Magenta
            (128, 0, 0),    # Bleu foncé
            (0, 128, 0),    # Vert foncé
            (0, 0, 128),    # Rouge foncé
            (128, 128, 0),  # Cyan foncé
            (0, 128, 128),  # Jaune foncé
            (128, 0, 128),  # Magenta foncé
        ]
        
        if class_id < len(colors):
            return colors[class_id]
        
        # Pour les classes supplémentaires, générer une couleur basée sur l'ID
        np.random.seed(class_id)
        return tuple(map(int, np.random.randint(0, 255, 3)))
        
    def get_class_id(self, class_name: str) -> Optional[int]:
        """
        Obtient l'ID de classe à partir du nom de classe.
        
        Args:
            class_name: Nom de la classe
            
        Returns:
            ID de la classe ou None si non trouvé
        """
        for cls_id, name in self.class_names.items():
            if name.lower() == class_name.lower():
                return cls_id
        return None
        
    def get_class_ids(self, class_names: List[str]) -> List[int]:
        """
        Obtient les IDs de classe à partir d'une liste de noms de classe.
        
        Args:
            class_names: Liste des noms de classe
            
        Returns:
            Liste des IDs de classe correspondants (ignore les noms non trouvés)
        """
        class_ids = []
        for name in class_names:
            cls_id = self.get_class_id(name)
            if cls_id is not None:
                class_ids.append(cls_id)
        return class_ids
