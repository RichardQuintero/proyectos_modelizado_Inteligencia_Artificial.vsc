import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Configuración visual de la app
st.set_page_config(page_title="Clasificador de Arquitectura", page_icon="🏛️", layout="centered")

st.title("🏛️ Clasificador de Estilos Arquitectónicos")
st.write("Sube una imagen exterior de un edificio para clasificar su estilo mediante **ResNet50**.")

# 1. Cargar el modelo guardado
@st.cache_resource
def load_trained_model():
    checkpoint = torch.load('modelo_arquitectura_resnet50.pth', map_location=torch.device('cpu'))
    classes = checkpoint['classes']
    
    # Reconstruir la arquitectura ResNet50
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, len(classes))
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, classes

try:
    model, classes = load_trained_model()
    st.success("Modelo cargado correctamente.")
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# Transformación idéntica a la fase de validación
inference_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 2. Subida de imagen
uploaded_file = st.file_uploader("Elige una foto (JPG, PNG)...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Imagen cargada", use_container_width=True)

    # Inferencia
    tensor = inference_transforms(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    top_prob, top_catid = torch.topk(probabilities, 1)
    estilo_ganador = classes[top_catid.item()]
    confianza = top_prob.item() * 100

    st.markdown("---")
    st.subheader(f"Estilo predicho: **{estilo_ganador.upper()}**")
    st.progress(confianza / 100.0)
    st.write(f"Confianza: **{confianza:.2f}%**")

    # Mostrar desglose por clase
    st.write("#### Probabilidades por categoría:")
    for i, class_name in enumerate(classes):
        st.write(f"- **{class_name.capitalize()}**: {probabilities[i].item()*100:.2f}%")