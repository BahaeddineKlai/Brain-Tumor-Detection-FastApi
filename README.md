# API de Classification des Tumeurs Cérébrales

Ce projet est une **API FastAPI** qui charge un modèle de deep learning entraîné et fournit un service de **classification d’images IRM du cerveau**.

L’API utilise un modèle **PyTorch** (EfficientNet-B0 ou ResNet50 en secours) et retourne :
- Le type de tumeur prédit
- Le niveau de confiance
- Le nom du fichier reçu

Le modèle fonctionne sur **CPU par défaut**.

---

## 🎯 Objectif du Projet

Ce backend permet de :
- Charger un modèle déjà entraîné
- Recevoir des images envoyées par l’utilisateur
- Exécuter une prédiction
- Retourner un résultat clair au frontend

Ce projet est destiné à être **téléchargé tel quel depuis GitHub** et exécuté localement.

---

## ⚙️ Installation

### 1. Cloner le projet

```bash
git clone <URL_DU_REPO>
cd <NOM_DU_DOSSIER>
```

### 2. Créer un environnement virtuel

Sous Windows :


```bash
python -m venv venv
venv\Scripts\activate
```


### 3. Installer les dépendances
```bash
pip install -r requirements.txt
```


▶️ Lancer le Serveur

Démarrer l’API avec :
````bash
uvicorn main:app --reload
````

L’API sera accessible sur :
````
http://127.0.0.1:8000
````
🔎 Endpoints Disponibles
Vérification de l’état du serveur

GET /

Réponse attendue :
````
{
  "message": "FastAPI Brain Tumor Classifier is running."
}
````
Prédiction d’Image
````
POST /predict
````
Paramètre requis (form-data) :

file → image à analyser

Exemple de réponse :
````
{
  "filename": "image_irm.jpg",
  "prediction": "glioma",
  "confidence": "92.30%",
  "confidence_value": 0.923
}
````