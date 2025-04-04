import pandas as pd
import re
import string
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore

stop_words = {
    'de','la','que','el','en','y','a','los','del','se','las','por','un','para','con','no','una','su','al','lo',
    'como','más','pero','sus','le','ya','o','este','sí','porque','esta','entre','cuando','muy','sin','sobre',
    'también','me','hasta','hay','donde','quien','desde','todo','nos','durante','todos','uno','les','ni','contra',
    'otros','ese','eso','ante','ellos','e','esto','mí','antes','algunos','qué','unos','yo','otro','otras','otra',
    'él','tanto','esa','estos','mucho','quienes','nada','muchos'
}

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\x00-\x7F]+', '', texto)
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    palabras = texto.split()
    return ' '.join([p for p in palabras if p not in stop_words and len(p) > 2])

palabras_ofensivas = {
    "idiota", "imbecil", "estupido", "basura", "asqueroso", "mierda", "miarda", "tonto", "inutil",
    "horrible", "inepto", "maldito", "puto", "puta", "pendejo", "naco", "loco", "pt", "perra", "perro"
}

def contiene_ofensas(texto):
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", "", texto)  # quitar puntuación
    palabras = texto.split()
    for palabra in palabras:
        if any(insulto in palabra for insulto in palabras_ofensivas):
            return True
    return False


def clasificar_sentimiento(row):
    if row['calificacion'] >= 4:
        return 'positivo'
    elif row['calificacion'] == 3:
        return 'neutro'
    else:
        return 'negativo'

df = pd.read_csv("Dataset_Comentarios_con_Calificacion.csv")
df['comentario_limpio'] = df['comentario'].apply(lambda x: limpiar_texto(str(x)))
df['sentimiento'] = df.apply(clasificar_sentimiento, axis=1)

positivos = df[df['sentimiento'] == 'positivo']
negativos = df[df['sentimiento'] == 'negativo']
neutros = df[df['sentimiento'] == 'neutro']

df_balanceado = pd.concat([
    positivos.sample(n=1000, replace=True, random_state=42),
    negativos.sample(n=1000, replace=True, random_state=42),
    neutros.sample(n=1000, replace=True, random_state=42)
], ignore_index=True).sample(frac=1, random_state=42)

X_text = df_balanceado['comentario_limpio'].values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_balanceado['sentimiento'])

X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)

tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)
X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)
X_train_pad = pad_sequences(X_train_seq, maxlen=50, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=50, padding='post')
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

modelo = Sequential([
    Embedding(5000, 64, input_length=50),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
modelo.fit(X_train_pad, y_train_cat, epochs=10, validation_data=(X_test_pad, y_test_cat), verbose=1)

print("\n🧪 Modelo entrenado. Escribe un comentario para analizar (o 'salir'):\n")
while True:
    entrada = input("Tu comentario: ")
    if entrada.lower() == "salir":
        break
    limpio = limpiar_texto(entrada)
    secuencia = tokenizer.texts_to_sequences([limpio])
    secuencia_pad = pad_sequences(secuencia, maxlen=50, padding='post')
    pred = modelo.predict(secuencia_pad, verbose=0)
    clase = label_encoder.inverse_transform([np.argmax(pred)])
    ofensivo = contiene_ofensas(entrada)
    print(f"➡️  Sentimiento: {clase[0].upper()}")
    print(f"🚫 Ofensivo: {'Sí' if ofensivo else 'No'}\n")


