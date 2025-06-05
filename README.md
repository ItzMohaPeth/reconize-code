# Système d'Analyse Vidéo Intelligent

Un système modulaire et extensible d'analyse vidéo utilisant YOLOv8 pour diverses tâches de détection et de comptage de personnes, ainsi que la détection de fumée et de feu.

## 🚀 Fonctionnalités

### 1. Comptage de Traversée de Ligne
- Détecte et compte les personnes qui traversent une ligne définie
- Indique la direction du passage (montant/descendant)
- Suivi en temps réel avec visualisation des trajectoires

### 2. Comptage Directionnel entre Zones
- Compte les personnes se déplaçant entre des zones prédéfinies
- Support de zones polygonales personnalisables
- Matrice de transition entre toutes les zones

### 3. Analyse du Temps d'Attente
- Identifie les files d'attente et mesure les temps d'attente
- Statistiques en temps réel (temps moyen, maximum)
- Alertes pour les temps d'attente excessifs

### 4. Détection d'Intrusion
- Alerte lors d'intrusions dans des zones interdites
- Support de zones multiples avec niveaux de sécurité
- Historique des alertes et notifications

### 5. Détection de Fumée et de Feu
- Détection basée sur YOLOv8 et analyse des couleurs
- Alertes d'urgence avec sauvegarde des preuves
- Surveillance de zones spécifiques

## 📋 Prérequis

- Python 3.8+
- Webcam ou fichiers vidéo pour les tests
- Modèle YOLOv8 (téléchargé automatiquement)

## 🛠️ Installation

1. **Cloner le projet**
\`\`\`bash
git clone <repository-url>
cd video-analysis-system
\`\`\`

2. **Installer les dépendances**
\`\`\`bash
pip install -r requirements.txt
\`\`\`

3. **Créer les dossiers nécessaires**
\`\`\`bash
mkdir -p models data results logs alerts/intrusion_snapshots alerts/fire_smoke_evidence
\`\`\`

4. **Télécharger le modèle YOLOv8** (optionnel - sera téléchargé automatiquement)
\`\`\`bash
# Le modèle sera téléchargé automatiquement lors de la première utilisation
# Vous pouvez aussi le télécharger manuellement :
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolov8n.pt
\`\`\`

## 🚀 Utilisation Rapide

### Script de Démonstration

Le moyen le plus simple de tester le système est d'utiliser le script de démonstration :

\`\`\`bash
# Comptage de traversée de ligne avec webcam
python scripts/demo.py line_crossing --source 0

# Comptage entre zones avec fichier vidéo
python scripts/demo.py zone_counting --source path/to/video.mp4

# Analyse du temps d'attente
python scripts/demo.py waiting_time --source 0

# Détection d'intrusion
python scripts/demo.py intrusion_detection --source 0

# Détection de fumée et feu
python scripts/demo.py fire_smoke_detection --source 0
\`\`\`

### Utilisation Avancée

Chaque fonctionnalité peut être lancée individuellement :

\`\`\`bash
# Comptage de traversée de ligne
python -m src.features.line_crossing.main --config configs/line_crossing.yaml --source 0

# Avec sauvegarde vidéo
python -m src.features.zone_counting.main --config configs/zone_counting.yaml --source video.mp4 --output output.mp4
\`\`\`

## ⚙️ Configuration

Chaque fonctionnalité dispose de son propre fichier de configuration YAML dans le dossier `configs/`.

### Exemple de Configuration - Traversée de Ligne

\`\`\`yaml
detector:
  model_path: "models/yolov8n.pt"
  conf_threshold: 0.25
  target_classes: ["person"]

line_crossing:
  line:
    start: [100, 300]  # Point de départ [x, y]
    end: [500, 300]    # Point d'arrivée [x, y]
\`\`\`

### Exemple de Configuration - Zones

\`\`\`yaml
zone_counting:
  zones:
    zone_a:
      name: "Entrée"
      points: [[50, 50], [200, 50], [200, 200], [50, 200]]
      color: [0, 255, 0]
    zone_b:
      name: "Sortie"
      points: [[300, 50], [450, 50], [450, 200], [300, 200]]
      color: [255, 0, 0]
\`\`\`

## 🏗️ Architecture du Projet

\`\`\`
src/
├── core/                    # Composants principaux
│   ├── video_processor.py   # Traitement vidéo
│   └── object_tracker.py    # Suivi d'objets
├── detector/                # Détection d'objets
│   └── yolov8_detector.py   # Interface YOLOv8
├── features/                # Fonctionnalités spécifiques
│   ├── line_crossing/       # Comptage de ligne
│   ├── zone_counting/       # Comptage de zones
│   ├── waiting_time/        # Temps d'attente
│   ├── intrusion_detection/ # Détection d'intrusion
│   └── fire_smoke_detection
