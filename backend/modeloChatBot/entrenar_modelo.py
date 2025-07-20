import os
import json
import re
import pickle
import fitz  # PyMuPDF: utilizado para leer y extraer texto de archivos PDF
from sklearn.feature_extraction.text import TfidfVectorizer  # Convierte texto en vectores numéricos
from sklearn.linear_model import LogisticRegression  # Algoritmo de clasificación supervisado
from sklearn.model_selection import train_test_split  # Divide los datos en entrenamiento y prueba
from sklearn.metrics import classification_report, confusion_matrix  # Métricas de evaluación
import matplotlib.pyplot as plt  # Visualización de gráficos
import seaborn as sns  # Mejora visual de los gráficos de matplotlib

# Función para extraer preguntas y respuestas desde archivos PDF
# Espera que el PDF esté estructurado con "Pregunta:" y "Respuesta:"
def extraer_faq_desde_pdf(ruta):
    faqs = []
    if not os.path.exists(ruta):
        return faqs
    doc = fitz.open(ruta)
    texto_total = ""
    for pagina in doc:
        texto_total += pagina.get_text()
    bloques = texto_total.split("Pregunta:")
    for bloque in bloques:
        if "Respuesta:" in bloque:
            partes = bloque.split("Respuesta:")
            pregunta = partes[0].strip()
            respuesta = partes[1].strip()
            respuesta = re.sub(r'\s*\d+\.\s$', '', respuesta)  # Elimina números al final como "92."
            if pregunta and respuesta:
                faqs.append((pregunta, respuesta))
    return faqs

# Detección de intención según ciertas palabras clave
# Asocia una intención a cada pregunta si se detectan palabras específicas
def detectar_intencion(pregunta):
    p = pregunta.lower()
    if any(x in p for x in ["precio", "cuánto cuesta", "valor", "costo", "cuánto vale", "cuánto sale", "cuánto es", "cuánto vale el curso", "cuánto cuesta el curso", "cuánto sale el curso", "cuánto cuesta la clase", "cuánto vale la clase", "cuánto sale la clase", "cuánto cuesta la sesión", "cuánto vale la sesión", "cuánto sale la sesión", "precio del curso", "precio de la clase", "precio de la sesión", "precio del taller", "precio del seminario", "precio del curso online", "precio del curso presencial", "precio del curso virtual", "precio del curso a distancia", "precio del curso en directo", "precio del curso en línea", "precio del curso por internet", "precio del curso por videoconferencia", "precio del curso por streaming", "precio del curso por webinar", "precio del curso por zoom", "precio del curso por meet", "precio del curso por teams", "precio del curso por skype", "precio del curso por hangouts", "precio de la clase online", "precio de la clase presencial"]):
        return "precio"
    if any(x in p for x in ["clase", "cuántas clases", "duración", "horas", "tiempo", "días", "horario", "clases", "sesiones", "clase en vivo", "clase grabada", "clase online", "clase presencial", "clase virtual", "clase a distancia", "clase en directo", "clase en línea", "clase por internet", "clase por videoconferencia", "clase por streaming", "clase por webinar", "clase por zoom", "clase por meet", "clase por teams", "clase por skype", "clase por hangouts", "clase por videollamada", "clase por videoconferencia", "clase por videollamada grupal", "clase por videollamada individual", "clase por videollamada en grupo", "clase por videollamada en vivo", "clase por videollamada en directo", "clase por videollamada en línea", "clase por videollamada a distancia", "clase por videollamada presencial", "clase por videollamada virtual", "clase por videollamada online", "clase por videollamada grabada", "clase por videollamada en diferido", "clase por videollamada en diferido grupal", "clase por videollamada en diferido individual"]):
        return "clases"
    if "certificado" in p:
        return "certificado"
    if any(x in p for x in ["contenido", "temario", "qué incluye", "detalles", "qué aprenderé", "qué veré", "qué temas", "qué aprenderás", "qué verás", "qué temas verás", "qué temas aprenderás", "qué temas incluye", "qué temas verás en el curso", "qué temas aprenderás en el curso", "qué temas incluye el curso", "qué temas verás en la clase", "qué temas aprenderás en la clase", "qué temas incluye la clase", "qué temas verás en la sesión", "qué temas aprenderás en la sesión", "qué temas incluye la sesión", "qué temas verás en el taller", "qué temas aprenderás en el taller", "qué temas incluye el taller", "qué temas verás en el seminario", "qué temas aprenderás en el seminario", "qué temas incluye el seminario", "qué temas verás en el curso online", "qué temas aprenderás en el curso online", "qué temas incluye el curso online", "qué temas verás en el curso presencial", "qué temas aprenderás en el curso presencial", "qué temas incluye el curso presencial"]):
        return "contenido"
    return None  # Si no se detecta ninguna intención

# Carga de configuración desde archivo externo JSON
# Permite modificar parámetros del modelo sin tocar el código
with open("config.json") as f:
    config = json.load(f)

# Extracción de datos desde todos los PDFs listados
faqs = []
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_DIR = os.path.join(BASE_DIR, "pdfs_para_entrenar")

faqs += extraer_faq_desde_pdf(os.path.join(PDF_DIR, "preguntas_respuestas_cursos_cocina.pdf"))
faqs += extraer_faq_desde_pdf(os.path.join(PDF_DIR, "contexto_extra_cursos_chatbot.pdf"))
faqs += extraer_faq_desde_pdf(os.path.join(PDF_DIR, "contexto_cursos_dos.pdf"))
faqs += extraer_faq_desde_pdf(os.path.join(PDF_DIR, "faq_chatbot_jemplos.pdf"))
faqs += extraer_faq_desde_pdf(os.path.join(PDF_DIR, "faq_chatbotpreguntas.pdf"))
faqs += extraer_faq_desde_pdf(os.path.join(PDF_DIR, "contexto_gemini.pdf"))

# Verificación de que se hayan encontrado preguntas y respuestas
if not faqs:
    raise ValueError("No se encontraron preguntas y respuestas en los PDFs")

# Filtrado de preguntas con intención válida y asociación a su respuesta
faqs_intencion = []
for pregunta, respuesta in faqs:
    intencion = detectar_intencion(pregunta)
    if intencion is not None:
        faqs_intencion.append((pregunta, intencion, respuesta))

# Preparación de datos para el entrenamiento
preguntas = [x[0] for x in faqs_intencion]  # Preguntas (features)
intenciones = [x[1] for x in faqs_intencion]  # Intenciones (labels)
respuestas_dict = {(x[1], x[0]): x[2] for x in faqs_intencion}  # Diccionario para mapear intenciones + preguntas => respuestas

# Guardar las respuestas en un archivo binario
with open("respuestas.pkl", "wb") as f:
    pickle.dump(respuestas_dict, f)

# Vectorización de texto con TF-IDF (mejora la representación del texto)
vectorizer = TfidfVectorizer(
    ngram_range=tuple(config["ngram_range"]),
    min_df=config["min_df"]
)
X = vectorizer.fit_transform(preguntas)

# Entrenamiento del modelo de clasificación con regresión logística
model = LogisticRegression(max_iter=config["max_iter"])
model.fit(X, intenciones)

# Guardar el modelo entrenado
with open("modelo_chatbot.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)

# Evaluación del modelo con métricas estándar
X_train, X_test, y_train, y_test = train_test_split(preguntas, intenciones, test_size=0.2, random_state=42)
X_train_vec = vectorizer.transform(X_train)
X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)

# Mostrar clasificación y matriz de confusión en consola y guardar gráfica
print(classification_report(y_test, y_pred))
labels = sorted(list(set(y_test)))
cm = confusion_matrix(y_test, y_pred, labels=labels)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Matriz de Confusión Chatbot")
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.tight_layout()
plt.savefig("matriz_confusion_chatbot.png")
plt.close()
print("Entrenamiento completo. Modelo y métricas guardadas.")