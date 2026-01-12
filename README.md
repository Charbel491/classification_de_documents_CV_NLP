
# 📄 Smart Doc Classifier (OCR + NLP)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-App-red) ![EasyOCR](https://img.shields.io/badge/OCR-EasyOCR-yellow) ![HuggingFace](https://img.shields.io/badge/NLP-SentenceTransformers-orange)

Une application intelligente de classification de documents administratifs (Factures, CNI, Relevés bancaires) combinant **Vision par Ordinateur** et **Traitement du Langage Naturel**.

Le projet utilise une approche hybride : **Règles métier strictes** (Mots-clés) prioritaires, avec un fallback sur une **analyse sémantique (Zero-Shot Classification)** via un LLM léger.

## 🚀 Fonctionnalités Clés

* **Pipeline Complet :** Conversion PDF/Image $\rightarrow$ Nettoyage OpenCV $\rightarrow$ OCR $\rightarrow$ Classification.
* **OCR Robuste :** Utilisation d'`EasyOCR` couplée à un pré-traitement d'image (Binarisation d'Otsu) pour lire des documents complexes, bruités ou bilingues (ex: Factures avec fond coloré).
* **Classification Hybride :**
  1. **Déterministe :** Recherche de mots-clés discriminants (ex: "IBAN", "Carte Nationale").
  2. **Sémantique :** Utilisation de `sentence-transformers` (HuggingFace) pour analyser le sens global du texte si les règles échouent.
* **Logique de Facturation Avancée :** Distinction intelligente entre facture d'Eau, d'Électricité ou Mixte au sein d'un même document.
* **Interface Utilisateur :** UI simple et interactive réalisée avec Streamlit.

## 📂 Classes de Documents Supportées

L'application classe automatiquement les pages dans les catégories suivantes :

* **Classe 1 : Facture d'eau et d'électricité** (Mixte)
  * *Sous-classe 1.1* : Facture d'eau uniquement
  * *Sous-classe 1.2* : Facture d'électricité uniquement
* **Classe 2 : CNI** (Carte Nationale d'Identité)
* **Classe 3 : Relevés bancaires**
* **Classe 4 : Autres** (Documents non identifiés)

## 🛠️ Architecture Technique

### Le Pipeline de Traitement

1. **Input :** L'utilisateur charge un PDF ou une image.
2. **Conversion & Zoom :** Les PDF sont convertis en images haute résolution (Zoom x2) via `PyMuPDF`.
3. **Pré-traitement (OpenCV) :**
   * Conversion en niveaux de gris.
   * **Binarisation (Thresholding)** pour supprimer les fonds colorés, les logos filigranés et le bruit.
4. **Extraction de Texte (OCR) :** `EasyOCR` extrait le texte brut.
5. **Classification (NLP) :**
   * Analyse des mots-clés présents.
   * Calcul d'embeddings (vecteurs de sens) et comparaison (Cosine Similarity) avec les définitions des classes.
6. **Output :** JSON structuré regroupant les pages par catégorie.

## 📦 Installation

### Pré-requis

* Python 3.9 ou supérieur
* Tesseract (Optionnel, non utilisé ici car EasyOCR est autonome)

### 1. Cloner le projet

```bash
git clone https://github.com/votre-user/smart-doc-classifier.git
cd smart-doc-classifier
```

### 2. Créer un environnement virtuel

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

**Contenu critique du `requirements.txt` :**

```text
numpy==1.26.4
opencv-python-headless<4.10.0
easyocr
streamlit
sentence-transformers
pymupdf
protobuf==3.20.3
Pillow
torch
torchvision
```

## ▶️ Utilisation

1) Générer d'abord un environnement virtuel

```bash
python -m venv venv
```

2) Activer l'environnement virtuel

```bash
.\venv\Scripts\activate
```

3) Installer les librairies

```bash
pip install -r requirements.txt
```

4) Lancez l'application Streamlit :

```bash
.\venv\Scripts\python -m streamlit run app.py
```

Ensuite :

1. Ouvrez votre navigateur sur l'URL indiquée (généralement `http://localhost:8501`).
2. Déposez un ou plusieurs fichiers (PDF, JPG, PNG).
3. Activez ou désactivez le **"Nettoyage d'image"** selon la qualité du document.
4. Cliquez sur **"Lancer l'analyse"**.
5. Récupérez le résultat au format JSON.

## 🧠 Modèles Utilisés

* **OCR Engine :** [EasyOCR](https://github.com/JaidedAI/EasyOCR) (Modèle Français).
* **NLP Model :** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
  * *Pourquoi ce modèle ?* Il est léger, rapide, supporte le multilingue et est très performant pour la similarité sémantique de phrases (Semantic Search).

## 📊 Exemple de Résultat JSON

```json
{
  "Classe 1 : facture d'eau et d'électricité": [
    "facture_janvier.pdf - Page 1"
  ],
  "Classe 2 : CNI": [
    "scan_cni.jpg"
  ],
  "Sous classe 1.2 : facture d'électricité": [
    "facture_janvier.pdf - Page 2"
  ]
}
```
