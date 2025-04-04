import pandas as pd
import re
import string

# Cargar tu archivo CSV
df = pd.read_csv("Comentarios_de_Clientes_sobre_Intelisis.csv")

# Lista de stopwords en español
stop_words = {
    'de', 'la', 'que', 'el', 'en', 'y', 'a', 'los', 'del', 'se', 'las', 'por', 'un',
    'para', 'con', 'no', 'una', 'su', 'al', 'lo', 'como', 'más', 'pero', 'sus', 'le',
    'ya', 'o', 'este', 'sí', 'porque', 'esta', 'entre', 'cuando', 'muy', 'sin',
    'sobre', 'también', 'me', 'hasta', 'hay', 'donde', 'quien', 'desde', 'todo', 'nos',
    'durante', 'todos', 'uno', 'les', 'ni', 'contra', 'otros', 'ese', 'eso', 'ante',
    'ellos', 'e', 'esto', 'mí', 'antes', 'algunos', 'qué', 'unos', 'yo', 'otro',
    'otras', 'otra', 'él', 'tanto', 'esa', 'estos', 'mucho', 'quienes', 'nada', 'muchos'
}

# Función para limpiar comentarios
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^\x00-\x7F]+', '', texto)
    texto = texto.translate(str.maketrans('', '', string.punctuation))
    palabras = texto.split()
    palabras_limpias = [p for p in palabras if p not in stop_words and len(p) > 2]
    return ' '.join(palabras_limpias)

# Aplicar limpieza
df['comentario_limpio'] = df['comentario'].apply(lambda x: limpiar_texto(str(x)))
