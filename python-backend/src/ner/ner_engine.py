"""
AgroVox - Module 4: Named Entity Recognition

STRATEGY
--------
A statistical/transformer NER model (spaCy `en_core_web_*` fine-tuned, or a
HF token-classification model) needs either a pretrained-weights download
(spaCy's small English pipeline is ~13MB from a PyPI-hosted wheel, which
*is* reachable in restricted environments that whitelist pypi.org - but many
lab/offline environments cannot reach even that). To guarantee the NER
module always works out of the box, AgroVox builds a spaCy `EntityRuler` on
a **blank** (no-download) `spacy.blank("en")` pipeline, driven by the
gazetteer/pattern list in data/ner_data.json (generated from the same
crop/disease/pest/fertilizer/soil vocabulary as the knowledge base). This is
a legitimate, standard NER technique (pattern/gazetteer-based NER) and is
fully deterministic/trainable-free, so it always runs offline.

If `en_core_web_sm` (or another spaCy pipeline with pretrained NER) is
available in the environment, `NEREngine` will layer the EntityRuler *on
top of* it so generic entities (e.g. PERSON, ORG) are still available
alongside the agricultural entity types - see `_build_pipeline()`.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

import spacy
from spacy.pipeline import EntityRuler  # noqa: F401 (ensures factory registration)


def _load_ner_data():
    with open(config.NER_JSON, encoding="utf-8") as f:
        return json.load(f)


class NEREngine:
    def __init__(self):
        self.data = _load_ner_data()
        self.nlp = self._build_pipeline()

    def _build_pipeline(self):
        base_model = None
        for candidate in ("en_core_web_sm",):
            try:
                base_model = spacy.load(candidate)
                break
            except Exception:
                continue

        nlp = base_model if base_model is not None else spacy.blank("en")

        if "entity_ruler" not in nlp.pipe_names:
            config_ = {"overwrite_ents": True}
            ruler = nlp.add_pipe("entity_ruler", config=config_,
                                  before="ner" if "ner" in nlp.pipe_names else None)
        else:
            ruler = nlp.get_pipe("entity_ruler")

        ruler.add_patterns(self.data["patterns"])
        return nlp

    def extract(self, text: str) -> list:
        """Return a list of {text, label, start, end} entity dicts."""
        if not text or not text.strip():
            return []
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            if ent.label_ in config.NER_LABELS:
                entities.append({
                    "text": ent.text, "label": ent.label_,
                    "start": ent.start_char, "end": ent.end_char,
                })
        return entities

    def extract_grouped(self, text: str) -> dict:
        """Group entities by label -> list of surface strings (deduplicated),
        convenient for the Query Analysis UI."""
        grouped = {}
        for ent in self.extract(text):
            grouped.setdefault(ent["label"], [])
            if ent["text"] not in grouped[ent["label"]]:
                grouped[ent["label"]].append(ent["text"])
        return grouped


if __name__ == "__main__":
    engine = NEREngine()
    text = "My rice crop has yellow leaves in Coimbatore, is it bacterial leaf blight?"
    for e in engine.extract(text):
        print(e)
    print(engine.extract_grouped(text))
