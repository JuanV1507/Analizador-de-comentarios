import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# === CONFIGURACIÓN ===
stop_words = {
    'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por', 'un',
    'para', 'con', 'no', 'una', 'su', 'al', 'lo', 'como', 'más', 'pero', 'sus', 'le',
    'ya', 'o', 'este', 'sí', 'porque', 'esta', 'entre', 'cuando', 'muy', 'sin',
    'sobre', 'también', 'me', 'hasta', 'hay', 'donde', 'quien', 'desde', 'todo', 'nos',
    'durante', 'todos', 'uno', 'les', 'ni', 'contra', 'otros', 'ese', 'eso', 'ante',
    'ellos', 'e', 'esto', 'mí', 'antes', 'algunos', 'qué', 'unos', 'yo', 'otro',
    'otras', 'otra', 'él', 'tanto', 'esa', 'estos', 'mucho', 'quienes', 'nada', 'muchos'
}

palabras_ofensivas = {
    "idiota", "imbecil", "estupido", "basura", "asqueroso", "mierda", "tonto", "inutil",
    "horrible", "inepto", "maldito", "puto", "puta", "pendejo", "naco", "loco"
}

# === FUNCIONES ===
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\x00-\x7F]+', '', texto)
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    palabras = texto.split()
    return ' '.join([p for p in palabras if p not in stop_words and len(p) > 2])

def contiene_ofensas(texto):
    palabras = texto.split()
    return any(p in palabras_ofensivas for p in palabras)

def clasificar_sentimiento(row):
    if row['calificacion'] >= 4:
        return 'positivo'
    elif row['calificacion'] == 3:
        return 'neutro'
    else:
        return 'negativo'

# === CARGA DE DATOS ===
df = pd.read_csv("Comentarios_de_Clientes_sobre_Intelisis1.csv")
df['comentario_limpio'] = df['comentario'].apply(lambda x: limpiar_texto(str(x)))
df['ofensivo'] = df['comentario_limpio'].apply(contiene_ofensas)
df['sentimiento_clasificado'] = df.apply(clasificar_sentimiento, axis=1)

# === ENTRENAMIENTO ===
X = df['comentario_limpio']
y = df['sentimiento_clasificado']

vectorizador = TfidfVectorizer()
X_vect = vectorizador.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vect, y, test_size=0.2, random_state=42)
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train, y_train)

print("✅ Modelo entrenado correctamente.")
print("🔎 Precisión en test set:")
print(classification_report(y_test, modelo.predict(X_test)))

# === PRUEBA EN VIVO ===
print("\n🧪 PRUEBA EN VIVO: escribe tu comentario (o escribe 'salir' para terminar)\n")
while True:
    entrada = input("Tu comentario: ")
    if entrada.lower() == "salir":
        break

    comentario_limpio = limpiar_texto(entrada)
    ofensivo = contiene_ofensas(comentario_limpio)
    comentario_vect = vectorizador.transform([comentario_limpio])
    prediccion = modelo.predict(comentario_vect)[0]

    print(f"➡️  Sentimiento: {prediccion.upper()}")
    print(f"🚫 Ofensivo: {'Sí' if ofensivo else 'No'}\n")
