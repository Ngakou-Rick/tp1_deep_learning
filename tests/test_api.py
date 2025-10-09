import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import requests

# Chargement du dataset MNIST pour obtenir une image de test
(_, _), (x_test, y_test) = keras.datasets.mnist.load_data()

# Prends la première image de test, par exemple (c'est un chiffre 7)
sample_image = x_test[0]
sample_label = y_test[0]

print(f"L'étiquette de l'image d'exemple est : {sample_label}")
# Normalise et aplatit l'image comme le ferait l'API
sample_image_processed = sample_image.astype("float32") / 255.0
sample_image_flattened = sample_image_processed.reshape(784).tolist()

# Sauvegarde dans un fichier JSON pour faciliter l'envoi
with open('mnist_sample_7.json', 'w') as f:
    json.dump({'image': sample_image_flattened}, f)

print("Image d'exemple MNIST (le chiffre 7) sauvegardée dans 'mnist_sample_7.json'")
print("Tu peux utiliser ce fichier avec le script test_api.py en remplaçant la ligne payload.")



url = 'http://127.0.0.1:5000/predict'

# Charger les données de l'image depuis le fichier JSON créé
with open('mnist_sample_7.json', 'r') as f:
    payload = json.load(f)

response = requests.post(url, json=payload)

if response.status_code == 200:
    print("Réponse de l'API :")
    print(response.json())
else:
    print(f"Erreur : {response.status_code}")
    print(response.json())