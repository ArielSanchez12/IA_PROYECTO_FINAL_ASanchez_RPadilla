from flask import Flask, request, jsonify
import pickle
import difflib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

with open("modelo_chatbot.pkl", "rb") as f:
    vectorizer, model = pickle.load(f)

with open("respuestas.pkl", "rb") as f:
    respuestas_dict = pickle.load(f)

# Cargar todas las preguntas e intenciones
faqs = []
for (intencion, pregunta) in [(k[0], k[1]) for k in respuestas_dict.keys()]:
    faqs.append((pregunta, intencion))

def detectar_curso(texto, lista_cursos):
    texto = texto.lower()
    for curso in lista_cursos:
        if curso in texto:
            return curso
    return None

@app.route("/preguntar", methods=["POST"])
def preguntar():
    data = request.get_json()
    texto = data.get("texto", "").lower()
    if not texto:
        return jsonify({"error": "No se recibió ninguna pregunta"}), 400

    # Respuestas fijas para saludo/despedida
    if "hola" in texto or "información" in texto:
        return jsonify({"respuesta": "¡Hola! Puedes preguntarme sobre precios, duración, contenido y requisitos de cualquiera de nuestros cursos."})
    if "gracias" in texto or "eso es todo" in texto:
        return jsonify({"respuesta": "¡Gracias por tu interés! Si tienes más preguntas sobre los cursos, aquí estaré."})

    texto_vec = vectorizer.transform([texto])
    intencion_pred = model.predict(texto_vec)[0]

    # Detectar cursos disponibles en las preguntas/respuestas
    cursos_disponibles = set()
    for p, i in faqs:
        for palabra in p.lower().split():
            if "curso" in palabra:
                cursos_disponibles.add(" ".join(p.lower().split()[-2:]))  # Ej: "comida americana"
    # O define manualmente:
    cursos_disponibles = [
        "curso de comida americana", "curso de comida mexicana", "curso de comida asiática",
        "curso de comida italiana", "curso de comida vegetariana", "curso de comida rápida"
    ]

    curso_usuario = detectar_curso(texto, cursos_disponibles)

    # Buscar la pregunta más parecida dentro de esa intención y ese curso
    preguntas_de_intencion = [
        p for p, i in faqs
        if i == intencion_pred and (curso_usuario is None or curso_usuario in p.lower())
    ]
    if not preguntas_de_intencion:
        return jsonify({"respuesta": "¿Puedes especificar el curso sobre el que quieres información?"})

    mejor_pregunta = None
    mejor_similitud = 0
    for p in preguntas_de_intencion:
        similitud = difflib.SequenceMatcher(None, texto, p.lower()).ratio()
        if similitud > mejor_similitud:
            mejor_similitud = similitud
            mejor_pregunta = p

    if mejor_similitud < 0.5:
        respuesta = "¿Puedes especificar tu pregunta sobre los cursos? Puedo ayudarte con precios, duración, contenido y requisitos."
    else:
        respuesta = respuestas_dict.get((intencion_pred, mejor_pregunta), "Lo siento, no tengo una respuesta para eso.")

    return jsonify({"respuesta": respuesta})

@app.route("/ping")
def ping():
    return "API del chatbot activa"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)