import os
import json
import re
import pickle
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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
            respuesta = re.sub(r'\s*\d+\.*\s*$', '', respuesta)
            if pregunta and respuesta:
                faqs.append((pregunta, respuesta))
    return faqs

def detectar_intencion(pregunta):
    p = pregunta.lower()
    if any(x in p for x in ["precio", "cuánto cuesta", "valor"]):
        return "precio"
    if any(x in p for x in ["clase", "cuántas clases", "duración"]):
        return "clases"
    if "certificado" in p:
        return "certificado"
    if any(x in p for x in ["contenido", "temario", "qué incluye"]):
        return "contenido"
    return None  # No uses "otro"

# Cargar parámetros
with open("config.json") as f:
    config = json.load(f)

# Extraer datos de todos los PDFs
faqs = []
faqs += extraer_faq_desde_pdf("./pdfs_para_entrenar/preguntas_respuestas_cursos_cocina.pdf")
faqs += extraer_faq_desde_pdf("./pdfs_para_entrenar/contexto_extra_cursos_chatbot.pdf")
faqs += extraer_faq_desde_pdf("./pdfs_para_entrenar/contexto_cursos_dos.pdf")
faqs += extraer_faq_desde_pdf("./pdfs_para_entrenar/faq_chatbot_jemplos.pdf")
faqs += extraer_faq_desde_pdf("./pdfs_para_entrenar/faq_chatbotpreguntas.pdf")
faqs += extraer_faq_desde_pdf("./pdfs_para_entrenar/contexto_gemini.pdf")
# Agrega aquí más PDFs si tienes

if not faqs:
    raise ValueError("No se encontraron preguntas y respuestas en los PDFs")

# Asignar intención a cada pregunta
faqs_intencion = []
for pregunta, respuesta in faqs:
    intencion = detectar_intencion(pregunta)
    if intencion is not None:
        faqs_intencion.append((pregunta, intencion, respuesta))
    # Si quieres saber cuántas se filtran, puedes imprimirlas:
    # else:
    #     print("Pregunta sin intención:", pregunta)

# Entrenamiento
preguntas = [x[0] for x in faqs_intencion]
intenciones = [x[1] for x in faqs_intencion]
respuestas_dict = {(x[1], x[0]): x[2] for x in faqs_intencion}

with open("respuestas.pkl", "wb") as f:
    pickle.dump(respuestas_dict, f)

vectorizer = TfidfVectorizer(
    ngram_range=tuple(config["ngram_range"]),
    min_df=config["min_df"]
)
X = vectorizer.fit_transform(preguntas)
model = LogisticRegression(max_iter=config["max_iter"])
model.fit(X, intenciones)

with open("modelo_chatbot.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)

# Métricas
X_train, X_test, y_train, y_test = train_test_split(preguntas, intenciones, test_size=0.2, random_state=42)
X_train_vec = vectorizer.transform(X_train)
X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)

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