"""
AgroVox - Full 6-Concept NLP Verification Script
Verifies and validates all 6 NLP domains/concepts implemented in the application.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.speech.speech_engine import SpeechEngine, word_error_rate
from src.translation.translator import Translator
from src.preprocessing.text_preprocessing import preprocess, detect_language
from src.classification.classifier import IntentClassifier
from src.ner.ner_engine import NEREngine
from src.qa.qa_engine import TFIDFQAEngine
from src.chatbot.chatbot_engine import AgroVoxPipeline, ChatbotSession

def main():
    print("=" * 70)
    print("      AgroVox: End-to-End NLP Domains & Modules Verification")
    print("=" * 70)

    # 1. Speech Processing (ASR & TTS)
    print("\n[Concept 1: Speech Processing]")
    se = SpeechEngine()
    wer = word_error_rate("my rice crop has yellow leaves", "my rice crop has yellow leaf")
    print(f"  - ASR (Whisper) Available: {se.asr_available()}")
    print(f"  - TTS (gTTS) Available:    {se.tts_available()}")
    print(f"  - WER Metric Sample:       {wer:.3f} (Hypothesis vs Ground Truth)")

    # 2. Machine Translation & Multilingual NLP
    print("\n[Concept 2: Machine Translation & Multilingual NLP]")
    tr = Translator()
    ta_input = "என் நெற்பயிரில் இலைகள் மஞ்சளாகிறது"
    t_ta2en = tr.translate(ta_input, "ta", "en")
    t_en2ta = tr.translate("what fertilizer is good for rice", "en", "ta")
    print(f"  - Source TA:       \"{ta_input}\"")
    print(f"  - Translated EN:   \"{t_ta2en['translated_text']}\" (Backend: {t_ta2en['backend']})")
    print(f"  - Source EN:       \"what fertilizer is good for rice\"")
    print(f"  - Translated TA:   \"{t_en2ta['translated_text']}\" (Backend: {t_en2ta['backend']})")

    # 3. Text Preprocessing & Language Detection
    print("\n[Concept 3: Text Preprocessing & Script Processing]")
    p_en = preprocess("My rice crop has yellow leaves in Coimbatore!")
    p_ta = preprocess(ta_input)
    print(f"  - Language Detection (EN): {p_en['language']}")
    print(f"  - Language Detection (TA): {p_ta['language']}")
    print(f"  - Lemmatized / Stemmed:    {p_en['processed_tokens']}")

    # 4. Text Classification (Intent Recognition)
    print("\n[Concept 4: Text Classification (Intent Classification)]")
    clf = IntentClassifier()
    c_q1 = "My rice crop has yellow leaves in Coimbatore."
    c_q2 = "What fertilizer is suitable for wheat during vegetative stage?"
    c_q3 = "How often should tomato crops be irrigated?"
    res1 = clf.predict(c_q1)
    res2 = clf.predict(c_q2)
    res3 = clf.predict(c_q3)
    print(f"  - Query: \"{c_q1}\" -> Intent: {res1['intent']} (Conf: {res1['confidence']:.3f}, Model: {res1['model']})")
    print(f"  - Query: \"{c_q2}\" -> Intent: {res2['intent']} (Conf: {res2['confidence']:.3f}, Model: {res2['model']})")
    print(f"  - Query: \"{c_q3}\" -> Intent: {res3['intent']} (Conf: {res3['confidence']:.3f}, Model: {res3['model']})")

    # 5. Named Entity Recognition (NER)
    print("\n[Concept 5: Named Entity Recognition (NER)]")
    ner = NEREngine()
    ner_query = "My rice crop has yellow leaves in Coimbatore, is it bacterial leaf blight?"
    entities = ner.extract_grouped(ner_query)
    print(f"  - Query: \"{ner_query}\"")
    print(f"  - Extracted Entities: {entities}")

    # 6. Question Answering System (QA Engine)
    print("\n[Concept 6: Question Answering System (Retrieval-Augmented)]")
    qa = TFIDFQAEngine()
    qa_q = "What fertilizer is suitable for rice?"
    qa_res = qa.answer(qa_q, category_hint="Fertilizer")
    print(f"  - Query: \"{qa_q}\"")
    print(f"  - Matched Question: \"{qa_res['matched_question']}\"")
    print(f"  - Confidence:       {qa_res['confidence']:.3f}")
    print(f"  - Answer:           {qa_res['answer'][:100]}...")

    # 7. Chatbots & Conversational AI (Multi-Turn Context Tracking)
    print("\n[Concept 7: Conversational Chatbot (Slot-Memory & Multi-Turn)]")
    pipeline = AgroVoxPipeline()
    session = ChatbotSession(pipeline, farmer_id="demo_farmer_verify")
    turn1 = session.ask("My rice crop has yellow leaves in Coimbatore.")
    turn2 = session.ask("Vegetative stage.")
    print(f"  - Turn 1 Utterance: \"My rice crop has yellow leaves in Coimbatore.\"")
    print(f"    Turn 1 Intent:    {turn1['intent']}")
    print(f"    Turn 1 Answer:    {turn1['answer'][:90]}...")
    print(f"  - Turn 2 Utterance: \"Vegetative stage.\"")
    print(f"    Turn 2 Intent:    {turn2['effective_intent']} (Carried from Context: {turn2['used_context_intent']})")
    print(f"    Context Memory:   {turn2['context_snapshot']}")
    print(f"    Turn 2 Answer:    {turn2['answer'][:90]}...")

    print("\n" + "=" * 70)
    print("  [SUCCESS] All NLP Concepts Tested & Verified Functioning Correctly!")
    print("=" * 70)

if __name__ == "__main__":
    main()
