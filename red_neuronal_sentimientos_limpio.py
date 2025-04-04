﻿
import pandas as pd
import re
import string
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# === CONFIGURACIÓN ===
stop_words = {
    'de','la','que','el','en','y','a','los','del','se','las','por','un','para','con','no','una','su','al','lo',
    'como','más','pero','sus','le','ya','o','este','sí','porque','esta','entre','cuando','muy','sin','sobre',
    'también','me','hasta','hay','donde','quien','desde','todo','nos','durante','todos','uno','les','ni','contra',
    'otros','ese','eso','ante','ellos','e','esto','mí','antes','algunos','qué','unos','yo','otro','otras','otra',
    'él','tanto','esa','estos','mucho','quienes','nada','muchos'
}

def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^ -]+', '', texto)
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    palabras = texto.split()
    return ' '.join([p for p in palabras if p not in stop_words and len(p) > 2])

# === CARGAR Y PREPROCESAR DATOS ===
df = pd.read_csv("Comentarios_de_Clientes_sobre_Intelisis.csv")
df['comentario_limpio'] = df['comentario'].apply(lambda x: limpiar_texto(str(x)))

def clasificar_sentimiento(row):
    if row['calificacion'] >= 4:
        return 'positivo'
    elif row['calificacion'] == 3:
        return 'neutro'
    else:
        return 'negativo'

df['sentimiento'] = df.apply(clasificar_sentimiento, axis=1)

# Balancear dataset
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

# === TOKENIZACIÓN Y PAD ===
vocab_size = 5000
max_length = 50

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)

X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

# === MODELO ===
modelo = Sequential([
    Embedding(vocab_size, 64, input_length=max_length),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])

modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
modelo.fit(X_train_pad, y_train_cat, epochs=10, validation_data=(X_test_pad, y_test_cat), verbose=1)

# === PRUEBA EN VIVO ===
print("\n✅ Modelo entrenado. Escribe un comentario (o 'salir') para analizar:")
while True:
    entrada = input("Tu comentario: ")
    if entrada.lower() == 'salir':
        break
    limpio = limpiar_texto(entrada)
    secuencia = tokenizer.texts_to_sequences([limpio])
    secuencia_pad = pad_sequences(secuencia, maxlen=max_length, padding='post')
    pred = modelo.predict(secuencia_pad, verbose=0)
    clase = label_encoder.inverse_transform([np.argmax(pred)])
    print(f"➡️  Sentimiento: {clase[0].upper()}\n")
