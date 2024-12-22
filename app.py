import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

def load_image():
    uploaded_file = st.file_uploader("Cargar imagen", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        return np.array(image)
    return None

def calculate_histogram(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(hist)
    ax.set_title('Histograma')
    ax.set_xlabel('Intensidad')
    ax.set_ylabel('Frecuencia')
    fig.tight_layout()
    return fig

def apply_threshold(image, threshold_value):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return binary

def apply_median_filter(image, kernel_size):
    return cv2.medianBlur(image, kernel_size)

def apply_gaussian_filter(image, kernel_size, sigma):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def apply_morphological_operations(image, operation, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if operation == 'Erosión':
        return cv2.erode(image, kernel, iterations=1)
    elif operation == 'Dilatación':
        return cv2.dilate(image, kernel, iterations=1)
    elif operation == 'Apertura':
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif operation == 'Cierre':
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return image

def main():
    st.set_page_config(layout="wide")
    st.title("Procesamiento de Imágenes NPT")
    
    # Configuración de la sidebar
    st.sidebar.header("Controles")
    
    # Cargar imagen
    image = load_image()
    
    if image is not None:
        # Contenedor para las imágenes procesadas
        processed_images = []
        processed_titles = []
        
        # Agregar imagen original
        processed_images.append(image)
        processed_titles.append("Imagen Original")
        
        # Histograma
        if st.sidebar.checkbox("Mostrar Histograma"):
            hist_fig = calculate_histogram(image)
            st.sidebar.pyplot(hist_fig)
        
        # Umbralización
        if st.sidebar.checkbox("Aplicar Umbralización"):
            threshold_value = st.sidebar.slider("Valor de umbral", 0, 255, 127)
            binary_image = apply_threshold(image, threshold_value)
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
            filtered_image = apply_median_filter(image, kernel_size)
            processed_images.append(filtered_image)
            processed_titles.append("Filtro de Mediana")
            image = filtered_image
            
        elif filter_type == "Gaussiano":
            kernel_size = st.sidebar.slider("Tamaño de kernel (Gaussiano)", 3, 9, 3, 2)
            sigma = st.sidebar.slider("Sigma", 0.1, 5.0, 1.0, 0.1)
            filtered_image = apply_gaussian_filter(image, kernel_size, sigma)
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
            morphed_image = apply_morphological_operations(image, morph_operation, kernel_size)
            processed_images.append(morphed_image)
            processed_titles.append(f'Operación {morph_operation}')
            image = morphed_image
        

        # Mostrar imágenes en una disposición de cuadrícula
        num_images = len(processed_images)
        images_per_row = 3
        num_rows = (num_images + images_per_row - 1) // images_per_row
        
        for row in range(num_rows):
            cols = st.columns(images_per_row)
            for col in range(images_per_row):
                idx = row * images_per_row + col
                if idx < num_images:
                    with cols[col]:
                        st.image(processed_images[idx], 
                                caption=processed_titles[idx],
                                use_column_width=True)
                        
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