import numpy as np
import matplotlib.pyplot as plt

# Créer des données d'exemple
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Créer un graphique
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-', linewidth=2, label='sin(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Graphique de la fonction sinus')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Graphique généré avec succès !")
print(f"Valeur maximale de y: {np.max(y):.2f}")
print(f"Valeur minimale de y: {np.min(y):.2f}")
