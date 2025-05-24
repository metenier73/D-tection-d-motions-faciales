
# 🎭 Détection d'Émotions Faciales avec Deep Learning

Ce projet détecte les émotions humaines à partir d’images de visages en utilisant un CNN entraîné sur le dataset FER2013.

## 📁 Structure du projet
emotion-detector/
├── model/
│   └── emotion_model.h5
├── static/images/
├── templates/
│   └── index.html
├── app.py
├── detect_emotion.py
├── requirements.txt
└── README.md

## 🧠 Objectif
- Classifier les émotions : Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Détection temps réel via OpenCV
- Déploiement web avec Flask

## 🚀 Utilisation
```bash
pip install -r requirements.txt
python app.py
```
Aller sur http://127.0.0.1:5000

## 👩‍💻 Auteur
Myriam Metenier – AI Engineer en formation
