import streamlit as st
import fitz  # PyMuPDF
import numpy as np
import easyocr
import cv2 
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import io

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Classification OCR", layout="wide")

@st.cache_resource
def load_models():
    """
    CORRECTION : On charge uniquement le Français ('fr').
    L'anglais est souvent inclus implicitement pour les chiffres/caractères.
    """
    # On retire 'ar' pour éviter le crash ValueError
    reader = easyocr.Reader(['fr'], gpu=False) 
    
    nlp_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    return reader, nlp_model

reader, nlp_model = load_models()

# -----------------------------------------------------------------------------
# PRÉ-TRAITEMENT (Nettoyage pour Factures)
# -----------------------------------------------------------------------------

def preprocess_image(img_np):
    """
    Convertit l'image en Noir & Blanc strict (Binarisation).
    Cela aide l'OCR à ignorer le fond coloré et les logos.
    """
    # 1. Niveaux de gris
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # 2. Binarisation (Otsu)
    # Tout ce qui n'est pas du texte noir devient blanc.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return binary

# -----------------------------------------------------------------------------
# CLASSIFICATION (Logique Métier)
# -----------------------------------------------------------------------------

def clean_text(text):
    return text.lower().replace('\n', ' ').strip()

def classify_page(text_content, model):
    text_clean = clean_text(text_content)
    
    # --- Mots-clés ---
    kw_cni =  [
        "carte nationale d'identité", "royaume du maroc", "n° d'état civil", 
        "valable jusqu'au", "n° de carte", "adresse", "fils de", "prénom", "nom", "fille de"
    ]
    kw_bank = [
       "relevé de compte bancaire", "extrait de compte", "situation de compte", "relevé d'identité bancaire rib", "solde au date valeur", "virement émis reçu", "retrait gab", "agios et commissions","attijariwafa bank", "banque populaire", "Al Barid", "cih bank", "bmce bank of africa", "société générale"
    ]
    
    # Factures
    kw_water = [
       "facture d’eau","eau potable","distribution d’eau","consommation d’eau","m³","volume consommé","index compteur eau","relevé compteur eau","abonnement eau","service des eaux","m3","index ancien / nouvel index","numéro compteur eau", "lydec", "redal", "amendis", "onee branchement eau", "SRM", "RADEEO", "RADEEF", "RADEET"
    ]
    kw_elec = [
    "facture d’électricité","énergie électrique","consommation électrique","kWh","kilowattheure","puissance souscrite","tension","index compteur électrique","relevé compteur électrique","électricité","kWh","ampère","phase","puissance","tarif électrique", 
        "lydec", "redal", "amendis", "onee branchement eau", "SRM", "RADEEO", "RADEEF", "RADEET"
    ]
    
    # 1. CNI
    cni_hits = sum(1 for w in kw_cni if w in text_clean)
    if "carte nationale" in text_clean or cni_hits >= 2:
        return "Classe 2 : CNI"

    # 2. Banque
    if any(k in text_clean for k in kw_bank):
        return "Classe 3 : relevés bancaires"

    # 3. Factures (Logique Eau/Elec)
    has_water = any(k in text_clean for k in kw_water)
    has_elec = any(k in text_clean for k in kw_elec)
    
    if has_water or has_elec:
        if has_water and has_elec:
            return "Classe 1 : facture d'eau et d'électricité"
        elif has_water:
            return "Sous classe 1.1 : facture d'eau"
        elif has_elec:
            return "Sous classe 1.2 : facture d'électricité"

    # --- Fallback Sémantique ---
    labels = [
        "facture eau électricité consommation énergie",
        "carte nationale identité document officiel",
        "relevé bancaire compte banque argent",
        "autre document administratif"
    ]
    label_map = {
        0: "Classe 1 : facture d'eau et d'électricité",
        1: "Classe 2 : CNI",
        2: "Classe 3 : relevés bancaires",
        3: "Classe 4 : Autres"
    }

    doc_emb = model.encode(text_content, convert_to_tensor=True)
    lbl_emb = model.encode(labels, convert_to_tensor=True)
    scores = util.cos_sim(doc_emb, lbl_emb)
    best_idx = int(np.argmax(scores.cpu().numpy()))
    predicted = label_map[best_idx]
    
    # Sécurité sous-classe
    if "Classe 1" in predicted:
        if has_water and not has_elec: return "Sous classe 1.1 : facture d'eau"
        if has_elec and not has_water: return "Sous classe 1.2 : facture d'électricité"
        
    return predicted

# -----------------------------------------------------------------------------
# TRAITEMENT PDF / IMAGE
# -----------------------------------------------------------------------------

def process_file(file_bytes, file_name, apply_preprocessing=True):
    extracted_data = []

    # PDF
    if file_name.lower().endswith('.pdf'):
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        for i, page in enumerate(doc):
            # Zoom x2 pour la netteté
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            
            # Nettoyage
            if apply_preprocessing:
                final_img = preprocess_image(img_np)
            else:
                final_img = img_np

            try:
                # detail=0 : texte brut
                # paragraph=True : combine les lignes proches
                results = reader.readtext(final_img, detail=0, paragraph=True)
                full_text = " ".join(results)
            except Exception as e:
                full_text = ""
                print(f"Erreur Page {i}: {e}")
            
            extracted_data.append((f"{file_name} - Page {i+1}", full_text))
            
    # IMAGES
    else:
        try:
            image = Image.open(io.BytesIO(file_bytes)).convert('RGB')
            img_np = np.array(image)
            
            if apply_preprocessing:
                final_img = preprocess_image(img_np)
            else:
                final_img = img_np
                
            results = reader.readtext(final_img, detail=0, paragraph=True)
            full_text = " ".join(results)
            extracted_data.append((f"{file_name}", full_text))
        except Exception:
            pass

    return extracted_data

# -----------------------------------------------------------------------------
# INTERFACE
# -----------------------------------------------------------------------------

st.title("📄 Classification EasyOCR + Nettoyage")

col1, col2 = st.columns(2)
with col1:
    use_prep = st.checkbox("Activer nettoyage (Noir & Blanc)", value=True, help="Recommandé pour supprimer les fonds colorés")
with col2:
    debug_mode = st.checkbox("Mode Debug (Voir texte lu)")

uploaded_files = st.file_uploader("Documents", accept_multiple_files=True)

if uploaded_files and st.button("Analyser"):
    
    results = {
        "Classe 1 : facture d'eau et d'électricité": [],
        "Sous classe 1.1 : facture d'eau": [],
        "Sous classe 1.2 : facture d'électricité": [],
        "Classe 2 : CNI": [],
        "Classe 3 : relevés bancaires": [],
        "Classe 4 : Autres": []
    }
    
    debug_logs = []
    bar = st.progress(0)
    
    for idx, file in enumerate(uploaded_files):
        pages = process_file(file.read(), file.name, apply_preprocessing=use_prep)
        
        for src, txt in pages:
            # Classification
            if not txt.strip():
                cat = "Classe 4 : Autres"
            else:
                cat = classify_page(txt, nlp_model)
                
            if cat in results: results[cat].append(src)
            else: results["Classe 4 : Autres"].append(src)
            
            if debug_mode:
                debug_logs.append(f"--- {src} ({cat}) ---\n{txt[:600]}...\n")
        
        bar.progress((idx+1)/len(uploaded_files))
        
    st.success("Terminé !")
    
    c_res, c_debug = st.columns(2)
    with c_res:
        st.subheader("Résultats")
        st.json({k: v for k, v in results.items() if v})
    with c_debug:
        if debug_mode:
            st.subheader("Texte brut")
            st.text_area("Log", "".join(debug_logs), height=600)