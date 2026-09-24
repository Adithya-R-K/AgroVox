"""
AgroVox - NER dataset generator (Phase 1/2 data + annotation).

Produces data/ner_data.json with two sections:

  "patterns"        -> gazetteer/pattern list used to build a spaCy
                        EntityRuler (a transparent, fully local, no-download
                        NER approach appropriate for a lab project - see
                        src/ner/ner_engine.py for the trainable-NER fallback
                        discussion).
  "eval_sentences"  -> hand-annotated (character-span) sentences used as a
                        held-out test set for computing entity-level
                        Precision / Recall / F1 (evaluation/ner_metrics.py).

Run: python data/generate_ner_data.py
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

CROPS = ["rice", "wheat", "maize", "cotton", "sugarcane", "groundnut",
         "tomato", "banana", "chilli", "blackgram"]
DISEASES = ["bacterial leaf blight", "rice blast", "late blight", "fusarium wilt",
            "powdery mildew", "panama wilt", "leaf curl virus", "blackgram yellow mosaic"]
SYMPTOMS = ["yellow leaves", "yellowing leaves", "wilting", "leaf curling",
            "brown spots", "white powdery growth", "stunted growth", "dead heart",
            "water-soaked lesions", "leaf blight"]
PESTS = ["brown planthopper", "stem borer", "whitefly", "fall armyworm",
         "pod borer", "thrips"]
FERTILIZERS = ["urea", "DAP", "MOP", "zinc sulphate", "NPK complex",
               "vermicompost", "single super phosphate", "farmyard manure"]
LOCATIONS = ["Coimbatore", "Madurai", "Thanjavur", "Salem", "Trichy", "Erode", "Tamil Nadu"]
SOIL_TYPES = ["black cotton soil", "red soil", "alluvial soil", "sandy loam",
              "clayey soil", "laterite soil"]
FARMING_ACTIVITIES = ["irrigation", "sowing", "harvesting", "transplanting",
                       "weeding", "spraying", "fertilizer application", "ploughing"]
WEATHER_CONDITIONS = ["heavy rain", "drought", "heatwave", "frost", "high humidity", "dry spell"]

PATTERNS = []
def add_patterns(label, terms):
    for t in terms:
        PATTERNS.append({"label": label, "pattern": t})
        PATTERNS.append({"label": label, "pattern": t.capitalize()})
        PATTERNS.append({"label": label, "pattern": t.title()})

add_patterns("CROP", CROPS)
add_patterns("DISEASE", DISEASES)
add_patterns("SYMPTOM", SYMPTOMS)
add_patterns("PEST", PESTS)
add_patterns("FERTILIZER", FERTILIZERS)
add_patterns("LOCATION", LOCATIONS)
add_patterns("SOIL_TYPE", SOIL_TYPES)
add_patterns("FARMING_ACTIVITY", FARMING_ACTIVITIES)
add_patterns("WEATHER_CONDITION", WEATHER_CONDITIONS)

# de-duplicate patterns (label, pattern) pairs
uniq = {}
for p in PATTERNS:
    uniq[(p["label"], p["pattern"])] = p
PATTERNS = list(uniq.values())


def find_span(text, sub):
    start = text.lower().find(sub.lower())
    if start == -1:
        raise ValueError(f"substring {sub!r} not found in {text!r}")
    return start, start + len(sub)


EVAL_RAW = [
    ("My rice crop has yellow leaves in Coimbatore.",
     [("rice", "CROP"), ("yellow leaves", "SYMPTOM"), ("Coimbatore", "LOCATION")]),
    ("The tomato field shows late blight symptoms after heavy rain.",
     [("tomato", "CROP"), ("late blight", "DISEASE"), ("heavy rain", "WEATHER_CONDITION")]),
    ("I applied urea to my cotton crop near Salem yesterday.",
     [("urea", "FERTILIZER"), ("cotton", "CROP"), ("Salem", "LOCATION")]),
    ("Whitefly is damaging the chilli plants in my field.",
     [("Whitefly", "PEST"), ("chilli", "CROP")]),
    ("The groundnut is growing in red soil with stunted growth this season.",
     [("groundnut", "CROP"), ("red soil", "SOIL_TYPE"), ("stunted growth", "SYMPTOM")]),
    ("We are doing irrigation for the sugarcane field in Thanjavur.",
     [("irrigation", "FARMING_ACTIVITY"), ("sugarcane", "CROP"), ("Thanjavur", "LOCATION")]),
    ("Bacterial leaf blight was found on rice leaves after a dry spell.",
     [("Bacterial leaf blight", "DISEASE"), ("rice", "CROP"), ("dry spell", "WEATHER_CONDITION")]),
    ("Banana plants in Madurai show wilting due to Panama wilt.",
     [("Banana", "CROP"), ("Madurai", "LOCATION"), ("wilting", "SYMPTOM"), ("Panama wilt", "DISEASE")]),
    ("Stem borer attack was noticed after spraying DAP on the maize crop.",
     [("Stem borer", "PEST"), ("DAP", "FERTILIZER"), ("maize", "CROP")]),
    ("Blackgram yellow mosaic is spreading in the blackgram field near Erode.",
     [("Blackgram yellow mosaic", "DISEASE"), ("blackgram", "CROP"), ("Erode", "LOCATION")]),
]

eval_sentences = []
for text, spans in EVAL_RAW:
    entities = []
    for sub, label in spans:
        s, e = find_span(text, sub)
        entities.append({"start": s, "end": e, "label": label, "text": text[s:e]})
    eval_sentences.append({"text": text, "entities": entities})


def main():
    payload = {
        "labels": config.NER_LABELS,
        "patterns": PATTERNS,
        "eval_sentences": eval_sentences,
    }
    with open(config.NER_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(PATTERNS)} gazetteer patterns and "
          f"{len(eval_sentences)} evaluation sentences -> {config.NER_JSON}")


if __name__ == "__main__":
    main()
