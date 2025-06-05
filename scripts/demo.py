#!/usr/bin/env python3
"""
Script de démonstration pour le système d'analyse vidéo intelligent.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Ajouter le répertoire src au chemin
sys.path.append(str(Path(__file__).parent.parent / "src"))

def run_feature(feature_name: str, config_path: str, source: str = "0", output: str = None):
    """
    Lance une fonctionnalité spécifique du système d'analyse vidéo.
    
    Args:
        feature_name: Nom de la fonctionnalité à lancer
        config_path: Chemin vers le fichier de configuration
        source: Source vidéo (webcam ou fichier)
        output: Chemin de sortie pour la vidéo (optionnel)
    """
    feature_modules = {
        "line_crossing": "src.features.line_crossing.main",
        "zone_counting": "src.features.zone_counting.main",
        "waiting_time": "src.features.waiting_time.main",
        "intrusion_detection": "src.features.intrusion_detection.main",
        "fire_smoke_detection": "src.features.fire_smoke_detection.main"
    }
    
    if feature_name not in feature_modules:
        print(f"Fonctionnalité '{feature_name}' non reconnue.")
        print(f"Fonctionnalités disponibles: {list(feature_modules.keys())}")
        return
        
    module_path = feature_modules[feature_name]
    
    # Construire la commande
    cmd = [
        sys.executable, "-m", module_path,
        "--config", config_path,
        "--source", source
    ]
    
    if output:
        cmd.extend(["--output", output])
        
    print(f"Lancement de {feature_name}...")
    print(f"Commande: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Erreur lors de l'exécution: {e}")
    except KeyboardInterrupt:
        print("\nArrêt demandé par l'utilisateur.")

def main():
    """Fonction principale du script de démonstration."""
    parser = argparse.ArgumentParser(
        description="Script de démonstration pour le système d'analyse vidéo intelligent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  python scripts/demo.py line_crossing --source 0
  python scripts/demo.py zone_counting --source video.mp4 --output output.mp4
  python scripts/demo.py intrusion_detection --config custom_config.yaml
        """
    )
    
    parser.add_argument(
        "feature",
        choices=["line_crossing", "zone_counting", "waiting_time", 
                "intrusion_detection", "fire_smoke_detection"],
        help="Fonctionnalité à démontrer"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Chemin vers le fichier de configuration (utilise le défaut si non spécifié)"
    )
    
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Source vidéo (0 pour webcam, chemin pour fichier vidéo)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Chemin pour sauvegarder la vidéo traitée"
    )
    
    args = parser.parse_args()
    
    # Déterminer le fichier de configuration par défaut
    if args.config is None:
        config_file = f"{args.feature}.yaml"
        args.config = f"configs/{config_file}"
        
    # Vérifier que le fichier de configuration existe
    if not os.path.exists(args.config):
        print(f"Fichier de configuration non trouvé: {args.config}")
        print("Assurez-vous que le fichier existe ou spécifiez un autre chemin avec --config")
        return
        
    # Lancer la fonctionnalité
    run_feature(args.feature, args.config, args.source, args.output)

if __name__ == "__main__":
    main()
