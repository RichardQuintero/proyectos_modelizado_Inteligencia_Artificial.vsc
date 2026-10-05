# Entrenamiento de un modelo de clasificación de estilos arquitectónicos utilizando ResNet50 preentrenada.
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

# 1. Configurar dispositivo (GPU si está disponible, sino CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo de entrenamiento: {device}")

# 2. Transformaciones y aumento de datos
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'dataset_arquitectura'
image_datasets = {
    x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
    for x in ['train', 'val']
}

# num_workers=0 evita problemas de hilos en Windows
dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=16, shuffle=(x == 'train'), num_workers=0)
    for x in ['train', 'val']
}

class_names = image_datasets['train'].classes
num_classes = len(class_names)
print(f"Clases a clasificar ({num_classes}): {class_names}")

# 3. Cargar ResNet50 preentrenada
print("\nCargando arquitectura ResNet50...")
weights = models.ResNet50_Weights.DEFAULT
model = models.resnet50(weights=weights)

# Congelar todas las capas convolucionales base
for param in model.parameters():
    param.requires_grad = False

# Reemplazar la capa fully connected (fc) de salida
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes)
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# 4. Bucle de entrenamiento
num_epochs = 8
inicio_tiempo = time.time()

print("\n--- Iniciando entrenamiento ---")

for epoch in range(num_epochs):
    print(f"\nÉpoca {epoch + 1}/{num_epochs}")
    print("-" * 25)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(image_datasets[phase])
        epoch_acc = running_corrects.double() / len(image_datasets[phase])

        print(f"[{phase.upper()}] Loss: {epoch_loss:.4f} | Acc: {epoch_acc * 100:.2f}%")

tiempo_total = time.time() - inicio_tiempo
print(f"\nEntrenamiento completado en {tiempo_total // 60:.0f}m {tiempo_total % 60:.0f}s")

# 5. Guardar modelo entrenado junto con sus etiquetas
archivo_salida = 'modelo_arquitectura_resnet50.pth'
torch.save({
    'model_state_dict': model.state_dict(),
    'classes': class_names
}, archivo_salida)

print(f"Modelo exportado exitosamente como: {archivo_salida}")