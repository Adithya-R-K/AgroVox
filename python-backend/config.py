"""
AgroVox - Global configuration
Central place for paths, constants and feature flags used across the project.
"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Directories -----------------------------------------------------
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

MODELS_DIR = os.path.join(BASE_DIR, "models")
CLASSIFIER_DIR = os.path.join(MODELS_DIR, "classifier")
NER_DIR = os.path.join(MODELS_DIR, "ner")
EMBEDDINGS_DIR = os.path.join(MODELS_DIR, "embeddings")
TRANSLATION_DIR = os.path.join(MODELS_DIR, "translation")

EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
EVAL_RESULTS_DIR = os.path.join(EVAL_DIR, "results")

DB_PATH = os.environ.get("AGROVOX_DB_PATH", os.path.join(BASE_DIR, "agrovox.db"))

for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, CLASSIFIER_DIR,
          NER_DIR, EMBEDDINGS_DIR, TRANSLATION_DIR, EVAL_DIR, EVAL_RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)
os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)

# ---- Dataset files -----------------------------------------------------
CROPS_CSV = os.path.join(DATA_DIR, "crops.csv")
DISEASES_CSV = os.path.join(DATA_DIR, "diseases.csv")
FERTILIZERS_CSV = os.path.join(DATA_DIR, "fertilizers.csv")
IRRIGATION_CSV = os.path.join(DATA_DIR, "irrigation.csv")
PESTS_CSV = os.path.join(DATA_DIR, "pests.csv")
SOIL_CSV = os.path.join(DATA_DIR, "soil.csv")
FAQ_CSV = os.path.join(DATA_DIR, "agriculture_faq.csv")
INTENTS_CSV = os.path.join(DATA_DIR, "intents.csv")
NER_JSON = os.path.join(DATA_DIR, "ner_data.json")
QA_CSV = os.path.join(DATA_DIR, "qa_dataset.csv")
TRANSLATIONS_CSV = os.path.join(DATA_DIR, "translations.csv")

# ---- Model artifact files ----------------------------------------------
CLASSIFIER_MODEL_PATH = os.path.join(CLASSIFIER_DIR, "best_classifier.joblib")
CLASSIFIER_VECTORIZER_PATH = os.path.join(CLASSIFIER_DIR, "tfidf_vectorizer.joblib")
CLASSIFIER_LABELS_PATH = os.path.join(CLASSIFIER_DIR, "label_encoder.joblib")
CLASSIFIER_COMPARISON_PATH = os.path.join(EVAL_RESULTS_DIR, "classifier_comparison.json")

NER_MODEL_DIR = os.path.join(NER_DIR, "agrovox_ner_model")

QA_VECTORIZER_PATH = os.path.join(EMBEDDINGS_DIR, "qa_tfidf_vectorizer.joblib")
QA_MATRIX_PATH = os.path.join(EMBEDDINGS_DIR, "qa_tfidf_matrix.joblib")

# ---- Intent / classification labels ------------------------------------
INTENT_LABELS = [
    "Crop Disease",
    "Fertilizer",
    "Irrigation",
    "Soil",
    "Pest Management",
    "Cultivation",
    "Weather",
    "Crop Information",
    "Harvesting",
    "General Agriculture",
]

# ---- NER entity labels ---------------------------------------------------
NER_LABELS = [
    "CROP", "DISEASE", "SYMPTOM", "PEST", "FERTILIZER",
    "LOCATION", "SOIL_TYPE", "FARMING_ACTIVITY", "WEATHER_CONDITION",
]

SUPPORTED_LANGUAGES = {"en": "English", "ta": "Tamil"}

RANDOM_SEED = 42

# ---- Environment-variable feature flags -------------------------------
# AgroVox runs fully offline out of the box (see README -> Fallback
# Strategy). These optional flags let a deployment with internet/GPU access
# opt in to the heavier neural backends without touching source code.
# Set them in a `.env` file (see `.env.example`) or your process
# environment / hosting platform's secrets manager - never hardcode keys.
ENABLE_NEURAL_MT = os.environ.get("AGROVOX_ENABLE_NEURAL_MT", "false").lower() == "true"
ENABLE_NEURAL_QA = os.environ.get("AGROVOX_ENABLE_NEURAL_QA", "false").lower() == "true"
ENABLE_WHISPER_ASR = os.environ.get("AGROVOX_ENABLE_WHISPER_ASR", "false").lower() == "true"
ENABLE_TTS = os.environ.get("AGROVOX_ENABLE_TTS", "false").lower() == "true"
# Optional: Hugging Face access token, only needed if a gated model is
# selected for the optional neural MT/QA backends above.
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "")

APP_TITLE = "🌾 AgroVox – Voice-Driven Intelligence for Smarter Farming"
DISCLAIMER = (
    "AgroVox is an agricultural information assistant. It is NOT a replacement "
    "for a certified agronomist or your local Agricultural Extension Office. "
    "For severe, uncertain, or economically significant crop problems, please "
    "verify with a qualified agricultural expert before taking action, "
    "especially before applying any pesticide or chemical."
)
