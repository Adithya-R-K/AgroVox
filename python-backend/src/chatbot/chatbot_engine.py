"""
AgroVox - Module 6: Conversational Chatbot + full pipeline orchestration.

This is the single integrated pipeline the whole spec insists on: one
AgroVoxPipeline object wires together Speech -> Language Detection ->
Translation -> Preprocessing -> Classification -> NER -> Knowledge
Retrieval -> QA -> Context-aware Chatbot response -> Translation back to
the farmer's language.

Context management: a lightweight slot-filling context tracker remembers
the most recently discussed crop/disease/symptom/growth-stage within a
conversation so obvious follow-up questions ("Vegetative stage.") get a
context-aware reply instead of being treated as a fresh, entity-less query.
"""
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config
from src.preprocessing.text_preprocessing import preprocess, detect_language
from src.translation.translator import Translator
from src.classification.classifier import IntentClassifier
from src.ner.ner_engine import NEREngine
from src.qa.qa_engine import get_qa_engine
from utils import db

GROWTH_STAGES = ["nursery", "seedling", "vegetative", "tillering", "flowering",
                  "fruiting", "maturity", "harvest"]


class ConversationContext:
    """Simple per-conversation slot memory (crop, disease, symptom, stage)."""

    def __init__(self):
        self.slots = {"CROP": None, "DISEASE": None, "SYMPTOM": None,
                       "LOCATION": None, "growth_stage": None, "last_intent": None}
        self.turns = []

    def update_from_entities(self, entities: dict):
        for label in ("CROP", "DISEASE", "SYMPTOM", "LOCATION"):
            values = entities.get(label)
            if values:
                self.slots[label] = values[0]

    def update_from_text(self, text: str):
        lowered = text.lower()
        for stage in GROWTH_STAGES:
            if stage in lowered:
                self.slots["growth_stage"] = stage
                break

    def as_context_string(self) -> str:
        parts = [f"{k}={v}" for k, v in self.slots.items() if v]
        return ", ".join(parts) if parts else "no prior context"


class AgroVoxPipeline:
    """The single integrated 6-concept NLP pipeline."""

    def __init__(self):
        self.translator = Translator()
        self.classifier = IntentClassifier()
        self.ner = NEREngine()
        self.qa = get_qa_engine()
        db.init_db()

    def process(self, raw_text: str, context: ConversationContext,
                preferred_output_language: str = None) -> dict:
        """Run the full pipeline on one farmer utterance and return a rich
        result dict used by both the UI and the DB logger."""
        # 1) language detection
        lang = detect_language(raw_text)
        output_lang = preferred_output_language or lang

        # 2) translation to English for internal NLP processing
        translation = self.translator.translate(raw_text, lang, "en")
        english_text = translation["translated_text"] if lang == "ta" else raw_text

        # 3) preprocessing
        pre = preprocess(english_text, language="en")

        # 4) text classification (intent)
        intent_result = self.classifier.predict(english_text, language="en")

        # 5) NER
        entities = self.ner.extract_grouped(english_text)

        # 6) update conversation context (follow-up handling)
        context.update_from_entities(entities)
        context.update_from_text(english_text)

        # If this turn is a short follow-up with no agricultural entities of
        # its own (e.g. just "vegetative stage" or "yes"), it is almost
        # certainly continuing the PREVIOUS topic rather than starting a new
        # one. In that case: (a) enrich the QA query with remembered slot
        # context, and (b) trust the conversation's last classified intent
        # for QA category-biasing rather than the classifier's isolated (and
        # often wrong, given how little text it has to work with) prediction
        # on this short utterance alone. This is what makes AgroVox a
        # genuine multi-turn chatbot rather than a stateless QA endpoint
        # that forgets the topic the moment a short reply arrives.
        is_short_followup = len(pre["tokens"]) <= 4 and not any(entities.values())
        query_for_qa = english_text
        if context.slots.get("CROP"):
            query_for_qa = f"{english_text} for {context.slots['CROP']} " \
                            f"({context.as_context_string()})"

        effective_intent = intent_result["intent"]
        if is_short_followup and context.slots.get("last_intent"):
            effective_intent = context.slots["last_intent"]
        context.slots["last_intent"] = effective_intent

        # 7) knowledge retrieval + question answering (biased toward the
        # effective intent, so the Classification + Context Management
        # modules genuinely inform the QA module in a single pipeline)
        qa_result = self.qa.answer(query_for_qa, category_hint=effective_intent)

        # 8) chatbot response composition (context-aware phrasing)
        answer_en = self._compose_chatbot_reply(english_text, effective_intent, entities,
                                                   context, qa_result)

        # 9) translate final answer back to farmer's preferred language
        final_translation = self.translator.translate(answer_en, "en", output_lang)
        answer_final = final_translation["translated_text"]

        return {
            "raw_text": raw_text,
            "detected_language": lang,
            "output_language": output_lang,
            "translated_input_en": english_text if lang == "ta" else None,
            "translation_backend": translation["backend"],
            "preprocessing": pre,
            "intent": intent_result["intent"],
            "intent_confidence": intent_result["confidence"],
            "classifier_model": intent_result["model"],
            "effective_intent": effective_intent,
            "used_context_intent": is_short_followup and effective_intent != intent_result["intent"],
            "entities": entities,
            "qa_matched_question": qa_result["matched_question"],
            "qa_source": qa_result["source"],
            "qa_confidence": qa_result["confidence"],
            "answer_en": answer_en,
            "answer": answer_final,
            "context_snapshot": dict(context.slots),
        }

    def _compose_chatbot_reply(self, english_text, effective_intent, entities,
                                context: ConversationContext, qa_result) -> str:
        """Adds a short conversational / follow-up framing around the raw
        QA answer, and asks a clarifying follow-up question when useful
        context (like growth stage) is still missing for a Crop Disease /
        Fertilizer query - this is what makes it a chatbot rather than a
        bare QA endpoint. `effective_intent` is the context-aware intent
        (see `process()`), not necessarily the classifier's raw prediction
        for this single utterance."""
        base_answer = qa_result["answer"]

        needs_stage = effective_intent in ("Crop Disease", "Fertilizer", "Irrigation")
        missing_stage = context.slots.get("growth_stage") is None

        reply = base_answer
        if needs_stage and missing_stage and qa_result["confidence"] < 0.5:
            reply += ("\n\nTo give you more targeted advice, could you tell me the current "
                      "growth stage of the crop (e.g. seedling, vegetative, flowering)?")
        return reply


class ChatbotSession:
    """Convenience wrapper the Streamlit app uses per browser session:
    owns a ConversationContext + conversation_id and logs every turn to
    SQLite, including feedback."""

    def __init__(self, pipeline: AgroVoxPipeline, farmer_id: str,
                 conversation_id: str = None):
        self.pipeline = pipeline
        self.farmer_id = farmer_id
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.context = ConversationContext()
        db.start_conversation(self.conversation_id, self.farmer_id)
        self.history = []

    def ask(self, text: str, input_mode: str = "text",
            preferred_output_language: str = None) -> dict:
        result = self.pipeline.process(text, self.context, preferred_output_language)
        query_id = db.log_query(
            conversation_id=self.conversation_id, farmer_id=self.farmer_id,
            raw_text=text, language=result["detected_language"],
            translated_text=result.get("translated_input_en"),
            intent=result["intent"], intent_confidence=result["intent_confidence"],
            entities=result["entities"], answer=result["answer"],
            answer_confidence=result["qa_confidence"], input_mode=input_mode)
        result["query_id"] = query_id
        self.history.append(result)
        return result

    def give_feedback(self, query_id: int, helpful: bool):
        db.log_feedback(query_id, "helpful" if helpful else "not_helpful")


if __name__ == "__main__":
    pipeline = AgroVoxPipeline()
    session = ChatbotSession(pipeline, farmer_id="demo_farmer")

    r1 = session.ask("My rice crop has yellow leaves in Coimbatore.")
    print("Farmer: My rice crop has yellow leaves in Coimbatore.")
    print("AgroVox:", r1["answer"])
    print("intent:", r1["intent"], "entities:", r1["entities"])
    print()

    r2 = session.ask("Vegetative stage.")
    print("Farmer: Vegetative stage.")
    print("AgroVox:", r2["answer"])
    print("context:", r2["context_snapshot"])
    print()

    r3 = session.ask("என் நெற்பயிரில் இலைகள் மஞ்சளாகிறது. என்ன செய்ய வேண்டும்?")
    print("Farmer (Tamil):", r3["raw_text"])
    print("Translated:", r3["translated_input_en"])
    print("AgroVox (Tamil):", r3["answer"])
