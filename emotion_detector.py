import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Charger le modèle entraîné
model = load_model('emotion_model.h5', compile=False)

# Charger le classifieur de visage de OpenCV
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Liste des émotions (doit correspondre au modèle)
emotion_labels = ['Colère', 'Dégoût', 'Peur', 'Heureux', 'Neutre', 'Triste', 'Surpris']

# Initialiser la webcam
cap = cv2.VideoCapture(0)

while True:
    # Lire une image de la webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir l’image en niveaux de gris
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les visages
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Dessiner un rectangle autour du visage
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Extraire la région du visage détecté
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        # Prétraiter pour le modèle
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # Prédire l’émotion
            preds = model.predict(roi)[0]
            label = emotion_labels[preds.argmax()]
            label_position = (x, y - 10)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Pas de visage détecté", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Afficher le résultat
    cv2.imshow('Détection d\'émotion', frame)

    # Quitter avec la touche 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
