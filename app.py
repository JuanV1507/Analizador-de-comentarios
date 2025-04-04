from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import re
import string

# === CARGA MODELO Y UTILIDADES ===
modelo = tf.keras.models.load_model("modelo.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

ofensivas_df = pd.read_csv("palabras_ofensivas.csv")
palabras_ofensivas = set(ofensivas_df["palabra"].dropna().str.lower())

# === APP FLASK ===
app = Flask(__name__)

stop_words = { ... }  # Pega aquí el mismo set de stopwords de antes

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^ -]+', '', texto)
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    palabras = texto.split()
    return ' '.join([p for p in palabras if p not in stop_words and len(p) > 2])

def contiene_ofensas(texto):
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", "", texto)
    palabras = texto.split()
    return any(p in palabras_ofensivas for p in palabras)

@app.route('/', methods=['GET', 'POST'])
def index():
    resultado = None
    ofensivo = None
    if request.method == 'POST':
        comentario = request.form['comentario']
        limpio = limpiar_texto(comentario)
        secuencia = tokenizer.texts_to_sequences([limpio])
        secuencia_pad = tf.keras.preprocessing.sequence.pad_sequences(secuencia, maxlen=100, padding='post')
        pred = modelo.predict(secuencia_pad, verbose=0)
        clase = label_encoder.inverse_transform([np.argmax(pred)])[0]
        ofensivo = contiene_ofensas(comentario)
        resultado = clase.upper()
    return render_template("index.html", resultado=resultado, ofensivo=ofensivo)

if __name__ == '__main__':
    app.run(debug=True)
