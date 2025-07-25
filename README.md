Cooking Courses 🍳 - Chatbot Educativo con Análisis de Sentimientos

Proyecto Final - Fundamentos de Inteligencia ArtificialTecnología Superior en Desarrollo de Software - Escuela Politécnica Nacional (ESFOT)Integrantes: Richard Padilla, Ariel Sánchez

🌐 Descripción del Proyecto

Cooking Courses es una aplicación educativa basada en inteligencia artificial que combina dos funcionalidades:

🤖 Un chatbot informativo que responde preguntas frecuentes sobre cursos de cocina.

📊 Un modelo de análisis de sentimientos que clasifica comentarios como positivos o negativos.

Ambos componentes están integrados en una interfaz visual desarrollada con HTML y respaldados por APIs en Flask.

💻 Tecnologías Usadas

Python 3.11+

HTML y CSS

Flask

Scikit-learn

Pandas / NLTK

PyMuPDF (para lectura de PDFs)

Git / GitHub

🚀 Instrucciones para Ejecutar el Sistema

1. Clonar el repositorio

git clone https://github.com/usuario/CookingCourses.git
cd CookingCourses

2. Crear entorno virtual y activar

python -m venv venv
source venv/bin/activate   # En Windows: venv\Scripts\activate

3. Instalar dependencias

pip install -r requirements.txt

4. Ejecutar las APIs (en terminales separadas)

API de Análisis de Sentimientos

cd backend/modeloOpiniones
python api_opiniones.py

API del Chatbot

cd backend/modeloChatBot
python api_chatbot.py

5. Ejecutar la interfaz (frontend)

cd frontend
streamlit run cursoscocina.html

🎨 Capturas del Sistema

Interfaz del chatbot:



Análisis de sentimiento:



Estructura del proyecto:



Matriz de confusión - Sentimientos



Matriz de confusión - Chatbot



🎥 Video Explicativo del Proyecto

Disponible en: https://youtu.be/ejemplo-enlace-videoEn este video se explica paso a paso el código, entrenamiento, integración y uso del sistema completo.

📚 Documentación

Puedes encontrar el informe técnico completo en la carpeta docs/ o como anexo en la plataforma de entrega.

🙌 Autores

Este proyecto fue desarrollado como parte de la asignatura Fundamentos de Inteligencia Artificial en la Escuela Politécnica Nacional:

Richard Padilla

Ariel Sánchez

🚧 Licencia

Este proyecto es académico. No se autoriza su uso con fines comerciales.

🔗 Repositorio Oficial

https://github.com/usuario/CookingCourses

✅ Gracias por probar Cooking Courses: aprende, pregunta y mejora tu experiencia educativa con IA 🌟

