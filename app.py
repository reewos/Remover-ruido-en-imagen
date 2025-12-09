import os
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

import utils

def load_image():
    uploaded_file = st.file_uploader("Cargar imagen", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        return np.array(image)
    return None

def load_sample_images(folder="images"):
    """Carga rutas de imágenes de ejemplo desde la carpeta especificada."""
    allowed = (".png", ".jpg", ".jpeg")
    if not os.path.exists(folder):
        return []

    return [os.path.join(folder, f) 
            for f in os.listdir(folder) 
            if f.lower().endswith(allowed)]

def main():
    st.set_page_config(layout="wide")
    st.title("Aplicación Interactiva para Remoción de Ruido en Imágenes")
    
    # Configuración de la sidebar
    st.sidebar.header("Controles")
    
    # Precargar imágenes de ejemplo
    sample_images = load_sample_images("images")
    st.sidebar.subheader("Imágenes de ejemplo")
    sample_choice = None
    if sample_images:
        filenames = [os.path.basename(img) for img in sample_images]
        sample_choice = st.sidebar.selectbox("Elegir imagen de muestra", ["Ninguna"] + filenames)

    # Cargar imagen
    image = load_image()

    # Si el usuario selecciona una imagen de ejemplo, reemplaza la imagen cargada
    if sample_choice and sample_choice != "Ninguna":
        idx = filenames.index(sample_choice)
        image = np.array(Image.open(sample_images[idx]).convert("RGB"))
        
    if image is not None:
        # Contenedor para las imágenes procesadas
        processed_images = []
        processed_titles = []
        
        # Agregar imagen original
        processed_images.append(image)
        processed_titles.append("Imagen Original")
        
        # Histograma
        if st.sidebar.checkbox("Mostrar Histograma"):
            hist_fig = utils.calculate_histogram(image)
            st.sidebar.pyplot(hist_fig)
        
        # Umbralización
        if st.sidebar.checkbox("Aplicar Umbralización"):
            threshold_value = st.sidebar.slider("Valor de umbral", 0, 255, 127)
            binary_image = utils.apply_threshold(image, threshold_value)
            processed_images.append(binary_image)
            processed_titles.append("Umbralización")
            image = binary_image
        
        # Filtros de ruido
        filter_type = st.sidebar.selectbox(
            "Seleccionar filtro de ruido",
            ["Ninguno", "Mediana", "Gaussiano"]
        )
        
        if filter_type == "Mediana":
            kernel_size = st.sidebar.slider("Tamaño de kernel (Mediana)", 3, 9, 3, 2)
            filtered_image = utils.apply_median_filter(image, kernel_size)
            processed_images.append(filtered_image)
            processed_titles.append("Filtro de Mediana")
            image = filtered_image
            
        elif filter_type == "Gaussiano":
            kernel_size = st.sidebar.slider("Tamaño de kernel (Gaussiano)", 3, 9, 3, 2)
            sigma = st.sidebar.slider("Sigma", 0.1, 5.0, 1.0, 0.1)
            filtered_image = utils.apply_gaussian_filter(image, kernel_size, sigma)
            processed_images.append(filtered_image)
            processed_titles.append("Filtro Gaussiano")
            image = filtered_image
        
        # Operaciones morfológicas
        morph_operation = st.sidebar.selectbox(
            "Operación morfológica",
            ["Ninguna", "Erosión", "Dilatación", "Apertura", "Cierre"]
        )
        
        if morph_operation != "Ninguna":
            kernel_size = st.sidebar.slider("Tamaño de kernel (Morfología)", 3, 9, 3, 2)
            morphed_image = utils.apply_morphological_operations(image, morph_operation, kernel_size)
            processed_images.append(morphed_image)
            processed_titles.append(f'Operación {morph_operation}')
            image = morphed_image
        

        # Mostrar imágenes en una disposición de cuadrícula
        num_images = len(processed_images)
        images_per_row = 4
        num_rows = (num_images + images_per_row - 1) // images_per_row
        
        for row in range(num_rows):
            cols = st.columns(images_per_row)
            for col in range(images_per_row):
                idx = row * images_per_row + col
                if idx < num_images:
                    with cols[col]:
                        st.image(processed_images[idx], 
                                caption=processed_titles[idx])
                        
        # Botón para descargar la imagen final
        if st.sidebar.button("Descargar imagen procesada"):
            final_image = Image.fromarray(image)
            buf = io.BytesIO()
            final_image.save(buf, format="PNG")
            btn = st.sidebar.download_button(
                label="Descargar",
                data=buf.getvalue(),
                file_name="imagen_procesada.png",
                mime="image/png"
            )

if __name__ == '__main__':
    main()