# Remover ruido en imagen

Este proyecto forma parte de la PEC1 del curso de Visión Artificial. El objetivo es demostrar diferentes técnicas de procesamiento digital de imágenes para eliminar ruido y mejorar la calidad visual.

**Autor:** Reewos Talla
 
## Características principales

La aplicación implementa de forma interactiva:

- Carga de imágenes propias o **imágenes de ejemplo** desde la carpeta `images/`.
- Visualización del **histograma de intensidades**.
- **Umbralización** con control del valor del umbral.
- **Filtros de reducción de ruido**:
  - Mediana  
  - Gaussiano
- **Operaciones morfológicas**:
  - Erosión  
  - Dilatación  
  - Apertura  
  - Cierre
- Visualización dinámica de cada etapa del procesamiento.
- Opción de **descargar la imagen final procesada**.

## Requisitos

Para ejecutar la aplicación localmente necesitas instalar las dependencias. Se recomienda crear un entorno virtual (opcional pero buena práctica):

```bash
python -m venv venv
source venv/bin/activate     # En Linux / macOS
venv\Scripts\activate        # En Windows
```

Instala las dependencias necesarias:

```bash
pip install -r requirements.txt
```

**Nota**: En caso de usar Streamlit Cloud, asegúrate de utilizar
opencv-python-headless en lugar de opencv-python, ya que evita problemas con dependencias gráficas.

## Ejecución en local

Lanza la aplicación con:

```bash
streamlit run app.py
```

Esto abrirá automáticamente la interfaz en tu navegador con la URL http://localhost:8501

## Uso en la aplicación

Dentro de la app se puede hacer lo siguiente:

1. **Cargar una imagen propia** (PNG/JPG/JPEG).
2. O seleccionar una imagen desde `images/` mediante el cargador de Streamlit.
3. Activar el **histograma**.
4. Aplicar un **umbral binario**.
5. Aplicar filtros de suavizado para **remover ruido**.
6. Aplicar operaciones morfológicas.
7. Ver el resultado en cuadrícula.
8. **Descargar la imagen final**
