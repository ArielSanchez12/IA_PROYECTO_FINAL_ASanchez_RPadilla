# PASO 1: config.json
# PASO 2: entrenar_modelo.py
import json
import pandas as pd
import pickle
import time
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import fitz  # PyMuPDF para leer PDFs

# ========== FUNCIONES AUXILIARES ==========
def leer_reseñas_pdf(ruta_pdf):
    reseñas = []
    if os.path.exists(ruta_pdf):
        doc = fitz.open(ruta_pdf)
        for page in doc:
            texto = page.get_text()
            for linea in texto.splitlines():
                if linea.strip():
                    reseñas.append(linea.strip())
    return reseñas

# ========== CARGAR CONFIGURACIÓN ==========
with open("config.json") as f:
    config = json.load(f)

# ========== MEDIR TIEMPO DE ENTRENAMIENTO ==========
inicio = time.time()

# ========== CARGAR DATOS DESDE HUGGING FACE ==========
print("Cargando dataset desde Hugging Face...")
dataset = load_dataset("tweet_eval", "sentiment")
textos = []
etiquetas = []

for ejemplo in dataset["train"]:
    if ejemplo["label"] != 1:  # 0 = negativa, 2 = positiva
        textos.append(ejemplo["text"])
        etiquetas.append(1 if ejemplo["label"] == 2 else 0)

dataset2 = load_dataset("imdb")
for ejemplo in dataset2["train"]:
    textos.append(ejemplo["text"])
    etiquetas.append(1 if ejemplo["label"] == 1 else 0)

# Dataset 3: yelp_review_full (solo 1 y 5 estrellas para polaridad)
dataset3 = load_dataset("yelp_review_full")
for ejemplo in dataset3["train"]:
    if ejemplo["label"] == 0:  # 1 estrella = negativo
        textos.append(ejemplo["text"])
        etiquetas.append(0)
    elif ejemplo["label"] == 4:  # 5 estrellas = positivo
        textos.append(ejemplo["text"])
        etiquetas.append(1)

# Dataset 4: amazon_polarity
dataset4 = load_dataset("amazon_polarity")
for ejemplo in dataset4["train"]:
    textos.append(ejemplo["content"])
    etiquetas.append(ejemplo["label"])  # 0 = negativo, 1 = positivo

# ========== CARGAR DATOS DESDE PDF (opcional) ==========
positivas_pdf = leer_reseñas_pdf("reseñas_positivas.pdf") + leer_reseñas_pdf("otras_positivas.pdf")
negativas_pdf = leer_reseñas_pdf("reseñas_negativas.pdf") + leer_reseñas_pdf("otras_negativas.pdf")

textos += positivas_pdf + negativas_pdf
etiquetas += [1] * len(positivas_pdf) + [0] * len(negativas_pdf)

# ========== CREAR DATAFRAME ==========
df = pd.DataFrame({"texto": textos, "etiqueta": etiquetas})
print(f"Total de reseñas cargadas: {len(df)}")

# ========== ENTRENAMIENTO ==========
X_train, X_test, y_train, y_test = train_test_split(df["texto"], df["etiqueta"], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(
    ngram_range=tuple(config["ngram_range"]),
    min_df=config["min_df"]
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=config["max_iter"])
model.fit(X_train_vec, y_train)

with open("modelo_sentimiento.pkl", "wb") as f:
    pickle.dump((vectorizer, model), f)
print("Modelo guardado como modelo_sentimiento.pkl")

# ========== MÉTRICAS ==========
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negativa", "Positiva"], yticklabels=["Negativa", "Positiva"])
plt.xlabel("Predicción")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.savefig("matriz_confusion.png")
plt.close()
print("Matriz de confusión guardada como matriz_confusion.png")
fin = time.time()
print(f"Tiempo total de entrenamiento y guardado: {fin - inicio:.2f} segundos")