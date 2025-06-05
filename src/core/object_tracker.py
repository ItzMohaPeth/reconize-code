import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import cv2
from scipy.optimize import linear_sum_assignment

class ObjectTracker:
    """
    Classe pour suivre les objets détectés à travers les frames.
    Implémentation simplifiée basée sur l'algorithme de suivi IoU.
    """
    def __init__(self, max_disappeared: int = 30, min_iou_threshold: float = 0.3):
        """
        Initialise le tracker d'objets.
        
        Args:
            max_disappeared: Nombre maximum de frames où un objet peut disparaître avant d'être supprimé
            min_iou_threshold: Seuil IoU minimum pour associer les détections aux objets suivis
        """
        self.next_object_id = 0
        self.objects = {}  # {object_id: {"bbox": [x1, y1, x2, y2], "class_id": class_id, ...}}
        self.disappeared = {}  # {object_id: count}
        self.max_disappeared = max_disappeared
        self.min_iou_threshold = min_iou_threshold
        self.history = {}  # {object_id: [{"bbox": [x1, y1, x2, y2], "frame": frame_idx}, ...]}
        self.current_frame = 0
        
    def register(self, detection: Dict[str, Any]) -> int:
        """
        Enregistre un nouvel objet.
        
        Args:
            detection: Dictionnaire contenant les informations de détection
            
        Returns:
            ID de l'objet enregistré
        """
        object_id = self.next_object_id
        self.objects[object_id] = detection
        self.disappeared[object_id] = 0
        self.history[object_id] = [{
            "bbox": detection["bbox"],
            "frame": self.current_frame,
            "centroid": self._get_centroid(detection["bbox"])
        }]
        self.next_object_id += 1
        return object_id
        
    def deregister(self, object_id: int):
        """
        Supprime un objet suivi.
        
        Args:
            object_id: ID de l'objet à supprimer
        """
        del self.objects[object_id]
        del self.disappeared[object_id]
        # Garder l'historique pour d'éventuelles analyses
        
    def update(self, detections: List[Dict[str, Any]], frame_idx: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Met à jour les objets suivis avec les nouvelles détections.
        
        Args:
            detections: Liste des détections dans la frame actuelle
            frame_idx: Index de la frame actuelle (optionnel)
            
        Returns:
            Liste des objets suivis avec leurs IDs
        """
        if frame_idx is not None:
            self.current_frame = frame_idx
        else:
            self.current_frame += 1
            
        # Si aucune détection, marquer tous les objets comme disparus
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
                    
            return []
            
        # Si aucun objet suivi, enregistrer toutes les détections
        if len(self.objects) == 0:
            for detection in detections:
                self.register(detection)
                
        # Sinon, associer les détections aux objets existants
        else:
            self._match_and_update(detections)
            
        # Préparer la liste des objets suivis à retourner
        tracked_objects = []
        for object_id, obj in self.objects.items():
            tracked_obj = obj.copy()
            tracked_obj["track_id"] = object_id
            tracked_obj["age"] = len(self.history[object_id])
            tracked_obj["time_since_update"] = self.disappeared[object_id]
            
            # Calculer la vitesse et la direction si possible
            if len(self.history[object_id]) > 1:
                current = self.history[object_id][-1]["centroid"]
                prev = self.history[object_id][-2]["centroid"]
                tracked_obj["velocity"] = (current[0] - prev[0], current[1] - prev[1])
            else:
                tracked_obj["velocity"] = (0, 0)
                
            tracked_objects.append(tracked_obj)
            
        return tracked_objects
        
    def _match_and_update(self, detections: List[Dict[str, Any]]):
        """
        Associe les détections aux objets existants et met à jour leur état.
        
        Args:
            detections: Liste des détections dans la frame actuelle
        """
        # Construire la matrice de coût (IoU négatif pour maximiser l'IoU)
        cost_matrix = np.zeros((len(self.objects), len(detections)))
        
        for i, (object_id, obj) in enumerate(self.objects.items()):
            for j, detection in enumerate(detections):
                # Calculer l'IoU entre l'objet suivi et la détection
                iou = self._calculate_iou(obj["bbox"], detection["bbox"])
                
                # Vérifier si la classe est la même
                same_class = obj["class_id"] == detection["class_id"]
                
                # Si la classe est différente ou l'IoU est trop faible, assigner un coût élevé
                if not same_class or iou < self.min_iou_threshold:
                    cost_matrix[i, j] = 1000  # Coût élevé pour éviter l'association
                else:
                    cost_matrix[i, j] = -iou  # Négatif car on veut maximiser l'IoU
                    
        # Résoudre le problème d'affectation
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Créer des ensembles pour suivre les objets et détections non associés
        used_objects = set()
        used_detections = set()
        
        # Mettre à jour les objets associés
        for row_idx, col_idx in zip(row_indices, col_indices):
            # Vérifier si l'association est valide (coût non prohibitif)
            if cost_matrix[row_idx, col_idx] < 1000:
                object_id = list(self.objects.keys())[row_idx]
                detection = detections[col_idx]
                
                # Mettre à jour l'objet avec la nouvelle détection
                self.objects[object_id] = detection
                self.disappeared[object_id] = 0
                
                # Ajouter à l'historique
                self.history[object_id].append({
                    "bbox": detection["bbox"],
                    "frame": self.current_frame,
                    "centroid": self._get_centroid(detection["bbox"])
                })
                
                # Marquer comme utilisés
                used_objects.add(object_id)
                used_detections.add(col_idx)
                
        # Gérer les objets non associés (disparus)
        for object_id in list(self.objects.keys()):
            if object_id not in used_objects:
                self.disappeared[object_id] += 1
                
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
                    
        # Enregistrer les nouvelles détections
        for i, detection in enumerate(detections):
            if i not in used_detections:
                self.register(detection)
                
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calcule l'IoU (Intersection over Union) entre deux boîtes englobantes.
        
        Args:
            box1: [x1, y1, x2, y2] de la première boîte
            box2: [x1, y1, x2, y2] de la deuxième boîte
            
        Returns:
            Valeur IoU entre 0 et 1
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculer les coordonnées de l'intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # Vérifier s'il y a une intersection
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
            
        # Calculer l'aire de l'intersection
        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculer l'aire des deux boîtes
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculer l'aire de l'union
        union_area = box1_area + box2_area - intersection_area
        
        # Calculer l'IoU
        iou = intersection_area / union_area if union_area > 0 else 0.0
        
        return iou
        
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
        
    def get_object_history(self, object_id: int) -> List[Dict[str, Any]]:
        """
        Récupère l'historique d'un objet suivi.
        
        Args:
            object_id: ID de l'objet
            
        Returns:
            Liste des positions de l'objet à travers les frames
        """
        if object_id in self.history:
            return self.history[object_id]
        return []
        
    def get_object_trajectory(self, object_id: int) -> List[Tuple[float, float]]:
        """
        Récupère la trajectoire d'un objet suivi (liste de centroïdes).
        
        Args:
            object_id: ID de l'objet
            
        Returns:
            Liste des centroïdes de l'objet à travers les frames
        """
        if object_id in self.history:
            return [entry["centroid"] for entry in self.history[object_id]]
        return []
