# Importamos las librerías necesarias
from flask import Flask, request, jsonify  # Flask para crear la API web
import pickle  # Para cargar los modelos y diccionarios ya entrenados
import difflib  # Para comparar similitud entre textos
from flask_cors import CORS  # Para permitir peticiones desde el frontend

# Inicializamos la aplicación Flask
app = Flask(__name__)
CORS(app)  # Habilita CORS para permitir peticiones desde cualquier origen

# Cargamos el modelo previamente entrenado (vectorizador TF-IDF y clasificador)
with open("modelo_chatbot.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

# Cargamos el diccionario de respuestas (clave: (intencion, pregunta), valor: respuesta)
with open("respuestas.pkl", "rb") as f:
    respuestas_dict = pickle.load(f)

# Extraemos todas las preguntas e intenciones del diccionario para facilitar el procesamiento
faqs = []
for (intencion, pregunta) in [(k[0], k[1]) for k in respuestas_dict.keys()]:
    faqs.append((pregunta, intencion))

# Función auxiliar para detectar si el texto contiene un curso específico
def detectar_curso(texto, lista_cursos):
    texto = texto.lower()
    for curso in lista_cursos:
        if curso in texto:
            return curso
    return None

# Endpoint principal que recibe las preguntas del usuario
@app.route("/preguntar", methods=["POST"])
def preguntar():
    data = request.get_json()
    texto = data.get("texto", "").lower()
    if not texto:
        return jsonify({"error": "No se recibió ninguna pregunta"}), 400

    # Respuestas fijas para saludos o cierres de conversación
    if "hola" in texto or "información" in texto:
        return jsonify({"respuesta": "¡Hola! Puedes preguntarme sobre precios, duración, contenido y requisitos de cualquiera de nuestros cursos."})
    if "gracias" in texto or "eso es todo" in texto:
        return jsonify({"respuesta": "¡Gracias por tu interés! Si tienes más preguntas sobre los cursos, aquí estaré."})

    # Convertimos el texto del usuario a vectores usando el vectorizador cargado
    texto_vec = vectorizer.transform([texto])
    # Predecimos la intención del texto usando el modelo de regresión logística
    intencion_pred = model.predict(texto_vec)[0]

    # Lista manual de los cursos disponibles para filtrar y contextualizar la respuesta
    cursos_disponibles = [
        "curso de comida americana", "curso de comida mexicana", "curso de comida asiática",
        "curso de comida italiana", "curso de comida vegetariana", "curso de comida rápida"
    ]
    curso_usuario = detectar_curso(texto, cursos_disponibles)  # Detecta si se menciona un curso

    # Filtramos las preguntas que coinciden con la intención y el curso mencionado
    preguntas_de_intencion = [
        p for p, i in faqs
        if i == intencion_pred and (curso_usuario is None or curso_usuario in p.lower())
    ]
    # Si no hay preguntas relacionadas con esa intención y curso
    if not preguntas_de_intencion:
        return jsonify({"respuesta": "¿Puedes especificar el curso sobre el que quieres información?"})

    # Buscamos la pregunta más parecida a la del usuario dentro de las filtradas
    mejor_pregunta = None
    mejor_similitud = 0
    for p in preguntas_de_intencion:
        similitud = difflib.SequenceMatcher(None, texto, p.lower()).ratio()
        if similitud > mejor_similitud:
            mejor_similitud = similitud
            mejor_pregunta = p

    # Si no hay suficiente similitud, devolvemos una respuesta genérica
    if mejor_similitud < 0.5:
        respuesta = "¿Puedes especificar tu pregunta sobre los cursos? Puedo ayudarte con precios, duración, contenido y requisitos."
    else:
        # Si hay buena similitud, buscamos la respuesta correspondiente
        respuesta = respuestas_dict.get((intencion_pred, mejor_pregunta), "Lo siento, no tengo una respuesta para eso.")

    # Devolvemos la respuesta en formato JSON
    return jsonify({"respuesta": respuesta})

# Endpoint de prueba para verificar si la API está corriendo correctamente
@app.route("/ping")
def ping():
    return "API del chatbot activa"

# Ejecutamos el servidor Flask si este archivo es el principal
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
