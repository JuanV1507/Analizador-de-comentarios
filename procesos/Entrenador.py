import random
from nltk import NaiveBayesClassifier, classify

# Función para convertir texto en features
def extract_features(texto):
    palabras = texto.split()
    return {palabra: True for palabra in palabras}

# Crear dataset para entrenamiento
dataset = [(extract_features(row['comentario_limpio']), row['sentimiento']) for _, row in df.iterrows()]
random.shuffle(dataset)

# Separar en entrenamiento y prueba (80/20)
train_size = int(len(dataset) * 0.8)
train_set, test_set = dataset[:train_size], dataset[train_size:]

# Entrenar modelo
modelo = NaiveBayesClassifier.train(train_set)

# Evaluar precisión
print("Precisión del modelo:", classify.accuracy(modelo, test_set))
modelo.show_most_informative_features(5)
