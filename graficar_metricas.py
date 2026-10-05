#Creación de un script para graficar las métricas de entrenamiento y validación
# de un modelo de clasificación de estilos arquitectónicos.
import matplotlib.pyplot as plt

# Métricas registradas durante el entrenamiento
epocas = list(range(1, 9))
train_acc = [75.00, 81.25, 85.42, 88.54, 90.62, 92.08, 93.75, 94.58]
val_acc   = [87.50, 89.17, 90.83, 91.67, 91.67, 92.50, 93.33, 93.33]

train_loss = [0.7597, 0.5210, 0.4105, 0.3340, 0.2780, 0.2315, 0.1980, 0.1720]
val_loss   = [0.4027, 0.3215, 0.2840, 0.2610, 0.2550, 0.2430, 0.2380, 0.2310]

plt.figure(figsize=(12, 5))

# Gráfica de Precisión (Accuracy)
plt.subplot(1, 2, 1)
plt.plot(epocas, train_acc, label='Train Acc', marker='o', color='#2ca02c')
plt.plot(epocas, val_acc, label='Val Acc', marker='s', color='#1f77b4')
plt.title('Evolución de la Precisión (Accuracy)')
plt.xlabel('Época')
plt.ylabel('Porcentaje (%)')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

# Gráfica de Pérdida (Loss)
plt.subplot(1, 2, 2)
plt.plot(epocas, train_loss, label='Train Loss', marker='o', color='#d62728')
plt.plot(epocas, val_loss, label='Val Loss', marker='s', color='#ff7f0e')
plt.title('Evolución de la Pérdida (Cross-Entropy Loss)')
plt.xlabel('Época')
plt.ylabel('Valor de Pérdida')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.tight_layout()
plt.savefig('curva_entrenamiento_resnet50.png', dpi=300)
print("Gráfica guardada exitosamente como 'curva_entrenamiento_resnet50.png'")