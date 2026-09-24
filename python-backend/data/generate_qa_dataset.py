"""
AgroVox - Question Answering dataset generator.

Builds data/qa_dataset.csv: a retrieval corpus of (question, answer,
category, source) rows derived from the knowledge-base CSVs (crops,
diseases, fertilizers, irrigation, pests, soil) plus the curated FAQ list.
This is the corpus the QA module (src/qa/qa_engine.py) indexes with TF-IDF
and retrieves from using cosine similarity.

Run (after generate_knowledge_base.py): python data/generate_qa_dataset.py
"""
import os
import sys
import csv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    rows = []

    # From FAQ (already Q/A pairs)
    for r in read_csv(config.FAQ_CSV):
        rows.append((r["question"], r["answer"], r["category"], "agriculture_faq.csv"))

    # From crops.csv
    for r in read_csv(config.CROPS_CSV):
        q = f"Tell me about growing {r['crop_name_en']}."
        a = (f"{r['crop_name_en']} ({r['crop_name_ta']}) is a {r['category'].lower()} crop typically "
             f"grown in the {r['season']} season, maturing in about {r['duration']}. "
             f"{r['description']} It prefers {r['soil_preference'].lower()} soil and a temperature "
             f"range of {r['temperature_range']}.")
        rows.append((q, a, "Crop Information", "crops.csv"))

    # From diseases.csv
    for r in read_csv(config.DISEASES_CSV):
        q = f"What are the symptoms of {r['disease_name_en']}?"
        a = (f"{r['disease_name_en']} ({r['disease_name_ta']}) affects {r['affected_crops']} and is "
             f"caused by {r['pathogen']}. Symptoms include: {r['symptoms']}.")
        rows.append((q, a, "Crop Disease", "diseases.csv"))
        q2 = f"How do I manage {r['disease_name_en']} in {r['affected_crops'].split(',')[0].strip()}?"
        a2 = f"Management of {r['disease_name_en']}: {r['management']}"
        rows.append((q2, a2, "Crop Disease", "diseases.csv"))

    # From fertilizers.csv
    for r in read_csv(config.FERTILIZERS_CSV):
        q = f"What fertilizer is suitable for {r['suitable_crops'].split(',')[0].strip()}?"
        a = (f"{r['fertilizer_name_en']} ({r['fertilizer_name_ta']}) supplies {r['nutrient_content']} "
             f"and is suitable for {r['suitable_crops']}. Recommended application: {r['application_stage']}. "
             f"{r['notes']}")
        rows.append((q, a, "Fertilizer", "fertilizers.csv"))

    # From irrigation.csv
    for r in read_csv(config.IRRIGATION_CSV):
        q = f"How often should {r['crop']} crops be irrigated?"
        a = (f"For {r['crop']}, the recommended method is {r['recommended_method']}. "
             f"{r['schedule_guideline']} {r['notes']}")
        rows.append((q, a, "Irrigation", "irrigation.csv"))

    # From pests.csv
    for r in read_csv(config.PESTS_CSV):
        q = f"How do I control {r['pest_name_en']} in {r['affected_crops'].split(',')[0].strip()}?"
        a = (f"{r['pest_name_en']} ({r['pest_name_ta']}) is a {r['pest_type'].lower()} affecting "
             f"{r['affected_crops']}. Damage symptoms: {r['damage_symptoms']}. Management: {r['management']}")
        rows.append((q, a, "Pest Management", "pests.csv"))

    # From soil.csv
    for r in read_csv(config.SOIL_CSV):
        q = f"Which crops grow well in {r['soil_name_en']}?"
        a = (f"{r['soil_name_en']} ({r['soil_name_ta']}) is characterized by: {r['characteristics']}. "
             f"It suits crops such as {r['suitable_crops']}. {r['notes']}")
        rows.append((q, a, "Soil", "soil.csv"))

    out_path = config.QA_CSV
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["question", "answer", "category", "source"])
        w.writerows(rows)

    print(f"Generated {len(rows)} QA pairs -> {out_path}")


if __name__ == "__main__":
    main()
