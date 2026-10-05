# Preparación de un Data set de imágenes para entrenamiento y validación de un modelo 
# de clasificación de estilos arquitectónicos.
import os
import shutil
import random

# Ruta base exacta a las carpetas con imágenes
base_estilos = r"C:\proyectos_modelizado_Inteligencia_Artificial\dataset_arquitectura\train"
carpeta_destino = "dataset_organizado"

# 4 estilos presentes en el dataset con alta diferenciación visual
estilos_seleccionados = {
    'gotico': 'gotico', 
    'barroco': 'barroco', 
    'bauhaus': 'bauhaus', 
    'egipcio': 'egipcio'
    }

print("Organizando imágenes en train y val...")

# Crear estructura de carpetas de destino
for split in ['train', 'val']:
    for etiqueta in estilos_seleccionados.values():
        os.makedirs(os.path.join(carpeta_destino, split, etiqueta), exist_ok=True)

random.seed(42)
total_imagenes = 0

for carpeta_origen, nombre_clase in estilos_seleccionados.items():
    ruta_clase = os.path.join(base_estilos, carpeta_origen)

    if not os.path.exists(ruta_clase):
        print(f"[!] Aviso: No se encontró la carpeta: {carpeta_origen}")
        continue

    archivos = [
        f for f in os.listdir(ruta_clase)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))
    ]
    random.shuffle(archivos)

    # 150 imágenes por clase (120 train / 30 val)
    seleccionados = archivos[:150]
    corte = int(len(seleccionados) * 0.8)

    train_archivos = seleccionados[:corte]
    val_archivos = seleccionados[corte:]

    for f in train_archivos:
        shutil.copy(os.path.join(ruta_clase, f), os.path.join(carpeta_destino, 'train', nombre_clase, f))
    for f in val_archivos:
        shutil.copy(os.path.join(ruta_clase, f), os.path.join(carpeta_destino, 'val', nombre_clase, f))

    print(f"[{nombre_clase.upper()}] Copiadas: {len(seleccionados)} -> Train: {len(train_archivos)} | Val: {len(val_archivos)}")
    total_imagenes += len(seleccionados)

print(f"\nListo: {total_imagenes} imágenes organizadas en '{carpeta_destino}'.")