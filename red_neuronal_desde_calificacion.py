
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

# === LIMPIEZA ===
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

def clasificar_sentimiento(row):
    if row['calificacion'] >= 4:
        return 'positivo'
    elif row['calificacion'] == 3:
        return 'neutro'
    else:
        return 'negativo'

# === CARGAR DATOS ===
df = pd.read_csv("Dataset_Comentarios_con_Calificacion.csv")
df['comentario_limpio'] = df['comentario'].apply(lambda x: limpiar_texto(str(x)))
df['sentimiento'] = df.apply(clasificar_sentimiento, axis=1)

# === TOKENIZACIÓN ===
X_text = df['comentario_limpio'].values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['sentimiento'])

X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)

tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train_text)
X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)

X_train_pad = pad_sequences(X_train_seq, maxlen=100, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=100, padding='post')
y_train_cat = to_categorical(y_train, num_classes=3)
y_test_cat = to_categorical(y_test, num_classes=3)

# === RED NEURONAL ===
modelo = Sequential([
    Embedding(10000, 64, input_length=100),
    GlobalAveragePooling1D(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(3, activation='softmax')
])
modelo.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
modelo.fit(X_train_pad, y_train_cat, epochs=10, validation_data=(X_test_pad, y_test_cat), verbose=1)

# === PRUEBA EN VIVO ===
print("\n✅ Modelo entrenado. Escribe un comentario para analizar (o 'salir'):")
while True:
    entrada = input("Tu comentario: ")
    if entrada.lower() == 'salir':
        break
    limpio = limpiar_texto(entrada)
    secuencia = tokenizer.texts_to_sequences([limpio])
    secuencia_pad = pad_sequences(secuencia, maxlen=100, padding='post')
    pred = modelo.predict(secuencia_pad, verbose=0)
    clase = label_encoder.inverse_transform([np.argmax(pred)])
    print(f"➡️  Sentimiento: {clase[0].upper()}\n")
