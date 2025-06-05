import cv2
import numpy as np
from typing import List, Dict, Any, Tuple, Optional

def draw_bounding_boxes(frame: np.ndarray, detections: List[Dict[str, Any]], 
                        color: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
    """
    Dessine les boîtes englobantes sur une frame.
    
    Args:
        frame: Image sur laquelle dessiner
        detections: Liste des détections
        color: Couleur à utiliser pour toutes les boîtes (None pour utiliser des couleurs par classe)
        
    Returns:
        Image avec les boîtes englobantes
    """
    result = frame.copy()
    
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox"])
        cls_id = det.get("class_id", 0)
        conf = det.get("confidence", 1.0)
        track_id = det.get("track_id", None)
        
        # Déterminer le label à afficher
        if track_id is not None:
            label = f"{det.get('class_name', 'Object')} #{track_id} {conf:.2f}"
        else:
            label = f"{det.get('class_name', 'Object')} {conf:.2f}"
            
        # Déterminer la couleur
        if color is None:
            # Générer une couleur basée sur l'ID de classe ou de suivi
            if track_id is not None:
                box_color = _get_color_by_id(track_id)
            else:
                box_color = _get_color_by_id(cls_id)
        else:
            box_color = color
            
        # Dessiner la boîte englobante
        cv2.rectangle(result, (x1, y1), (x2, y2), box_color, 2)
        
        # Dessiner l'étiquette
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(result, (x1, y1 - text_size[1] - 5), (x1 + text_size[0], y1), box_color, -1)
        cv2.putText(result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
    return result

def draw_trajectories(frame: np.ndarray, tracker: Any, max_history: int = 30) -> np.ndarray:
    """
    Dessine les trajectoires des objets suivis.
    
    Args:
        frame: Image sur laquelle dessiner
        tracker: Instance du tracker d'objets
        max_history: Nombre maximum de points d'historique à dessiner
        
    Returns:
        Image avec les trajectoires
    """
    result = frame.copy()
    
    for object_id in tracker.objects:
        # Récupérer la trajectoire
        trajectory = tracker.get_object_trajectory(object_id)
        
        # Limiter l'historique
        if len(trajectory) > max_history:
            trajectory = trajectory[-max_history:]
            
        # Dessiner la trajectoire
        if len(trajectory) > 1:
            color = _get_color_by_id(object_id)
            
            # Dessiner les lignes entre les points
            for i in range(1, len(trajectory)):
                pt1 = tuple(map(int, trajectory[i-1]))
                pt2 = tuple(map(int, trajectory[i]))
                thickness = 2
                cv2.line(result, pt1, pt2, color, thickness)
                
            # Dessiner le dernier point plus grand
            cv2.circle(result, tuple(map(int, trajectory[-1])), 5, color, -1)
            
    return result

def draw_line(frame: np.ndarray, start_point: Tuple[int, int], end_point: Tuple[int, int], 
              color: Tuple[int, int, int] = (0, 0, 255), thickness: int = 2) -> np.ndarray:
    """
    Dessine une ligne sur une frame.
    
    Args:
        frame: Image sur laquelle dessiner
        start_point: Point de départ (x, y)
        end_point: Point d'arrivée (x, y)
        color: Couleur de la ligne
        thickness: Épaisseur de la ligne
        
    Returns:
        Image avec la ligne
    """
    result = frame.copy()
    cv2.line(result, start_point, end_point, color, thickness)
    return result

def draw_polygon(frame: np.ndarray, points: List[Tuple[int, int]], 
                 color: Tuple[int, int, int] = (0, 0, 255), thickness: int = 2, 
                 fill: bool = False) -> np.ndarray:
    """
    Dessine un polygone sur une frame.
    
    Args:
        frame: Image sur laquelle dessiner
        points: Liste des points du polygone [(x1, y1), (x2, y2), ...]
        color: Couleur du polygone
        thickness: Épaisseur des lignes
        fill: Si True, remplit le polygone
        
    Returns:
        Image avec le polygone
    """
    result = frame.copy()
    points_array = np.array(points, np.int32)
    points_array = points_array.reshape((-1, 1, 2))
    
    if fill:
        # Créer un masque transparent pour le remplissage
        overlay = result.copy()
        cv2.fillPoly(overlay, [points_array], color)
        
        # Fusionner avec l'image originale avec transparence
        alpha = 0.4  # Transparence
        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
        
        # Dessiner le contour
        cv2.polylines(result, [points_array], True, color, thickness)
    else:
        cv2.polylines(result, [points_array], True, color, thickness)
        
    return result

def draw_text(frame: np.ndarray, text: str, position: Tuple[int, int], 
              font_scale: float = 1.0, color: Tuple[int, int, int] = (255, 255, 255), 
              thickness: int = 2, background: bool = True) -> np.ndarray:
    """
    Dessine du texte sur une frame.
    
    Args:
        frame: Image sur laquelle dessiner
        text: Texte à afficher
        position: Position (x, y) du texte
        font_scale: Échelle de la police
        color: Couleur du texte
        thickness: Épaisseur du texte
        background: Si True, ajoute un fond au texte
        
    Returns:
        Image avec le texte
    """
    result = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    if background:
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        x, y = position
        cv2.rectangle(result, (x, y - text_size[1] - 5), (x + text_size[0], y + 5), (0, 0, 0), -1)
        
    cv2.putText(result, text, position, font, font_scale, color, thickness)
    return result

def draw_counter(frame: np.ndarray, label: str, count: int, position: Tuple[int, int], 
                 font_scale: float = 0.8, color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    Dessine un compteur sur une frame.
    
    Args:
        frame: Image sur laquelle dessiner
        label: Étiquette du compteur
        count: Valeur du compteur
        position: Position (x, y) du compteur
        font_scale: Échelle de la police
        color: Couleur du texte
        
    Returns:
        Image avec le compteur
    """
    result = frame.copy()
    text = f"{label}: {count}"
    
    # Dessiner un fond semi-transparent
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
    x, y = position
    
    # Fond du compteur
    cv2.rectangle(result, (x - 5, y - text_size[1] - 10), (x + text_size[0] + 5, y + 5), (0, 0, 0), -1)
    
    # Texte du compteur
    cv2.putText(result, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2)
    
    return result

def draw_alert(frame: np.ndarray, text: str, position: Optional[Tuple[int, int]] = None, 
               font_scale: float = 1.2, blink: bool = True, frame_count: int = 0) -> np.ndarray:
    """
    Dessine une alerte sur une frame.
    
    Args:
        frame: Image sur laquelle dessiner
        text: Texte de l'alerte
        position: Position (x, y) de l'alerte (None pour centrer)
        font_scale: Échelle de la police
        blink: Si True, fait clignoter l'alerte
        frame_count: Compteur de frames pour le clignotement
        
    Returns:
        Image avec l'alerte
    """
    result = frame.copy()
    
    # Déterminer si l'alerte doit être visible (clignotement)
    visible = True
    if blink:
        visible = (frame_count // 15) % 2 == 0  # Clignote toutes les 15 frames
        
    if visible:
        # Calculer la position si non spécifiée
        if position is None:
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            position = ((result.shape[1] - text_size[0]) // 2, (result.shape[0] + text_size[1]) // 2)
            
        # Dessiner un fond rouge semi-transparent
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
        x, y = position
        
        # Fond de l'alerte
        cv2.rectangle(result, (x - 10, y - text_size[1] - 10), (x + text_size[0] + 10, y + 10), (0, 0, 255), -1)
        
        # Texte de l'alerte
        cv2.putText(result, text, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
        
    return result

def _get_color_by_id(id_value: int) -> Tuple[int, int, int]:
    """
    Génère une couleur unique basée sur un ID.
    
    Args:
        id_value: Valeur de l'ID
        
    Returns:
        Tuple (B, G, R) représentant la couleur
    """
    # Liste de couleurs prédéfinies pour les premiers IDs
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
    
    if id_value < len(colors):
        return colors[id_value]
    
    # Pour les IDs supplémentaires, générer une couleur basée sur l'ID
    np.random.seed(id_value)
    return tuple(map(int, np.random.randint(0, 255, 3)))
