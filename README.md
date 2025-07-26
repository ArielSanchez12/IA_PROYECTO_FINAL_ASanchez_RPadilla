# 🍳 COOKING COURSES  
## Chatbot Educativo con Análisis de Sentimientos

---

### Proyecto Final  
**Fundamentos de Inteligencia Artificial**  
Tecnología Superior en Desarrollo de Software  
Escuela Politécnica Nacional (ESFOT)

**Integrantes:**  
- Richard Padilla  
- Ariel Sánchez

---

## 🌐 Descripción del Proyecto

**Cooking Courses** es una aplicación educativa basada en inteligencia artificial que combina dos funcionalidades principales:

1. 🤖 **Chatbot informativo:** Responde preguntas frecuentes sobre cursos de cocina.  
2. 📊 **Análisis de sentimientos:** Clasifica comentarios como positivos o negativos para retroalimentar la experiencia del usuario.

Ambos componentes están integrados en una interfaz visual desarrollada con HTML y respaldados por APIs implementadas con Flask.

---

## 💻 Tecnologías Usadas

- Python 3.11+  
- HTML y CSS  
- Flask  
- Scikit-learn  
- Pandas / NLTK  
- PyMuPDF (para lectura de PDFs)  
- Git / GitHub  

---

## Instrucciones para Ejecutar el Sistema 🚀

### 1. Clonar el repositorio  

git clone https://github.com/ArielSanchez12/IA_PROYECTO_FINAL_ASanchez_RPadilla.git

### 2. Instalar dependencias

pip install scikit-learn matplotlib seaborn datasets pymupdf tqdm pandas

### 3.- Entrenar modelos (en terminales separadas)

- Analisis de sentimientos

cd backend/modeloOpiniones:
python entrenar_modelo.py

- Chatbot

cd backend/modeloChatbot:
python entrenar_modelo.py

`⚠️ Esto puede demorar dependiendo de cuántos datos cargados esté entrenando, así que tomará tiempo.`

### 3. Ejecutar las APIs (en terminales separadas)

- API de Análisis de Sentimientos

cd backend/modeloOpiniones:
python api_opiniones.py

- API del Chatbot

cd backend/modeloChatBot:
python api_chatbot.py

### 4. Ejecutar la interfaz (frontend)

cd frontend/cursoscocina.html y ejecutamos con open with `Live Server`.

---------------------------------------------------------------------------------------------------------------

## 🎨 Capturas del Sistema

- Interfaz del chatbot:

<img width="453" height="768" alt="image" src="https://github.com/user-attachments/assets/26496bbb-95f8-48f6-99f2-9f9e6c06c367" />

---

- Análisis de sentimiento:

<img width="1896" height="521" alt="image" src="https://github.com/user-attachments/assets/b5924126-e8e2-476f-b786-d36ecee19a81" />

---

- Estructura del proyecto:

<img width="317" height="756" alt="image" src="https://github.com/user-attachments/assets/06c1e4f6-0544-438d-840d-3bcf76f9a5a5" />

---

- Matriz de confusión - Sentimientos

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/0e3e76d3-6318-4fff-a695-ee30f64b3a74" />

---

- Matriz de confusión - Chatbot

<img width="640" height="480" alt="image" src="https://github.com/user-attachments/assets/b074428b-f774-464a-9023-1b4fa04d1d7c" />

---

## 📦 Despliegue del Proyecto 

### ✅ Opción 1: Ejecución Local

Dirígete a la sección 👉 [Instrucciones para Ejecutar el Sistema](#instrucciones-para-ejecutar-el-sistema-)

### ✅ Opción 2: Despliegue en la Web

Enlace: https://proyectofinal-ia-asanchez-rpadilla.netlify.app 

`⚠️ Puede demorarse 1 o 2 minutos en activarse las APIs debido al plan gratis de Render`

---

## 🎥 Video Explicativo del Proyecto

Disponible en: https://youtu.be/8hOE3q9i1Hk 

En este video se explica paso a paso el código, entrenamiento, integración y uso del sistema completo.

---

## 📚 Documentación

Puedes ver el archivo completo en este [📄 PDF del proyecto](./DOCUMENTACION%20DEL%20PROYECTO%20FINAL%20-%20RPadilla%20-%20ASanchez.pdf).

---

## 🧠 Presentacion

Disponible en: https://www.canva.com/design/DAGuOL1uLpw/r90Rrw3jdXPz8uAO89t8OQ/view?utm_content=DAGuOL1uLpw&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h3b9fa76d59

En esta presentacion se hablara sobre nuestros objetivos con este proyecto y como funciona el sistema.

---


✅ Gracias por probar Cooking Courses: aprende, pregunta y mejora tu experiencia educativa con IA 🌟


