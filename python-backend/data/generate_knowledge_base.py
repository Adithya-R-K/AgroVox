"""
AgroVox - Knowledge base generator (Phase 1: Data Collection)

Generates the core agricultural knowledge-base CSV files that power the
Question Answering and Knowledge Explorer modules. The agricultural facts
below are standard, widely-published extension-service knowledge (rice,
wheat, cotton, maize, sugarcane, groundnut, tomato, banana, chilli,
blackgram) and are written to be informational, not prescriptive of exact
chemical dosages.

Run:  python data/generate_knowledge_base.py
"""
import os
import csv
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  wrote {path}  ({len(rows)} rows)")


# ---------------------------------------------------------------- CROPS --
CROPS = [
    ("Rice", "நெல்", "Cereal", "Kharif/Rabi", "90-150 days",
     "Clayey loam, water-retentive", "20-35°C",
     "Staple cereal grown widely in Tamil Nadu deltas; needs standing water for most growth stages."),
    ("Wheat", "கோதுமை", "Cereal", "Rabi", "110-130 days",
     "Well-drained loam", "10-25°C",
     "Major cereal in North India; sensitive to waterlogging and needs a cool growing period."),
    ("Maize", "சோளம்", "Cereal", "Kharif/Rabi", "90-110 days",
     "Well-drained sandy loam", "18-27°C",
     "Versatile cereal used for food, fodder and industry; moderate water requirement."),
    ("Cotton", "பருத்தி", "Fibre", "Kharif", "150-180 days",
     "Black cotton soil (regur)", "21-30°C",
     "Fibre crop needing a long frost-free season and moderate, well-spaced irrigation."),
    ("Sugarcane", "கரும்பு", "Cash crop", "Year-round (12 month crop)", "300-365 days",
     "Deep, well-drained loam", "21-27°C",
     "Long-duration cash crop with high water demand, propagated via stem cuttings."),
    ("Groundnut", "நிலக்கடலை", "Oilseed/Legume", "Kharif/Rabi", "100-130 days",
     "Well-drained sandy loam", "20-30°C",
     "Legume oilseed that fixes atmospheric nitrogen and needs calcium during pegging."),
    ("Tomato", "தக்காளி", "Vegetable", "Year-round (season dependent)", "60-90 days",
     "Well-drained sandy loam", "20-27°C",
     "Widely grown vegetable, sensitive to blossom-end rot and several fungal/bacterial diseases."),
    ("Banana", "வாழை", "Fruit", "Year-round", "10-13 months",
     "Deep, rich, well-drained loam", "15-35°C",
     "High water and nutrient demanding fruit crop, propagated using suckers/tissue-culture plants."),
    ("Chilli", "மிளகாய்", "Vegetable/Spice", "Kharif/Rabi", "120-180 days",
     "Well-drained sandy loam", "20-30°C",
     "Spice crop sensitive to thrips, mites and viral leaf curl disease."),
    ("Blackgram", "உளுந்து", "Pulse/Legume", "Kharif/Rabi", "70-90 days",
     "Well-drained loam to clay loam", "25-35°C",
     "Short-duration pulse commonly grown as a rice-fallow crop; improves soil nitrogen."),
]
CROPS_HEADER = ["crop_id", "crop_name_en", "crop_name_ta", "category", "season",
                "duration", "soil_preference", "temperature_range", "description"]
crops_rows = [(f"C{idx+1:03d}", *c) for idx, c in enumerate(CROPS)]

# ------------------------------------------------------------- DISEASES --
DISEASES = [
    ("Bacterial Leaf Blight", "பாக்டீரியா இலை கருகல் நோய்", "Rice", "Xanthomonas oryzae",
     "Water-soaked lesions on leaf margins turning yellow-white and drying from tip inward",
     "Use resistant varieties, avoid excess nitrogen, ensure field sanitation and balanced water management; consult an expert for approved bactericides."),
    ("Rice Blast", "நெல் வெடிப்பு நோய்", "Rice", "Magnaporthe oryzae (fungus)",
     "Diamond/spindle shaped grey-centered lesions with brown margins on leaves, neck rot at panicle base",
     "Use resistant varieties, avoid dense nursery, balanced nitrogen; fungicidal seed treatment as advised by an agricultural officer."),
    ("Late Blight", "பிற்பாதி கருகல் நோய்", "Tomato", "Phytophthora infestans (oomycete)",
     "Dark water-soaked patches on leaves and stems with white fungal growth under humid conditions, fruit rot",
     "Improve field drainage and air circulation, remove infected debris, use certified disease-free seedlings; fungicide only under expert guidance."),
    ("Fusarium Wilt", "பியூசேரியம் வாட்டநோய்", "Cotton, Tomato, Banana", "Fusarium oxysporum (fungus)",
     "Yellowing and wilting of lower leaves progressing upward, vascular browning inside stem",
     "Practice crop rotation, use resistant/tolerant varieties, avoid waterlogging, solarize nursery soil before sowing."),
    ("Powdery Mildew", "பொடி பூஞ்சை நோய்", "Groundnut, Chilli, many crops", "Erysiphe / Oidium spp. (fungus)",
     "White powdery fungal growth on upper leaf surface, leaf curling and premature drying",
     "Ensure good airflow between plants, avoid excess nitrogen, remove heavily infected leaves; sulfur-based fungicide only per label/expert advice."),
    ("Panama Wilt", "பனாமா வாட்டநோய்", "Banana", "Fusarium oxysporum f. sp. cubense",
     "Yellowing of older leaves, splitting of pseudostem base, plant collapse",
     "Use tissue-culture disease-free planting material, avoid waterlogged soil, do not replant bananas in infected fields for several years."),
    ("Leaf Curl Virus", "இலை சுருள் வைரஸ் நோய்", "Chilli, Tomato", "Begomovirus (whitefly transmitted)",
     "Upward/downward curling of leaves, stunted growth, reduced fruiting",
     "Control whitefly vector with sticky traps and recommended practices, remove infected plants early, use virus-tolerant varieties."),
    ("Blackgram Yellow Mosaic", "மஞ்சள் தேமல் நோய்", "Blackgram, Greengram", "Mungbean yellow mosaic virus (whitefly transmitted)",
     "Irregular yellow mosaic patches on leaves, stunted pods",
     "Sow tolerant varieties, manage whitefly populations, remove volunteer/alternate host plants near field borders."),
]
DISEASES_HEADER = ["disease_id", "disease_name_en", "disease_name_ta", "affected_crops",
                    "pathogen", "symptoms", "management"]
diseases_rows = [(f"D{idx+1:03d}", *d) for idx, d in enumerate(DISEASES)]

# ----------------------------------------------------------- FERTILIZERS --
FERTILIZERS = [
    ("Urea", "யூரியா", "Nitrogen (N) - 46%", "Rice, Maize, Wheat, most cereals",
     "Vegetative growth stage, split doses",
     "Fast-acting nitrogen source promoting leafy growth; apply in split doses to reduce leaching loss."),
    ("DAP (Di-Ammonium Phosphate)", "டிஏபி உரம்", "Nitrogen (18%) + Phosphorus (46%)",
     "Most crops at sowing/transplanting", "Basal application at sowing/transplanting",
     "Provides early-stage phosphorus for root development, usually applied as a basal dose."),
    ("MOP (Muriate of Potash)", "மியூரியேட் ஆஃப் பொட்டாஷ்", "Potassium (K) - 60%",
     "Fruiting/tuber crops, sugarcane, banana", "Flowering / fruit development stage",
     "Improves fruit quality, disease resistance and drought tolerance; important during reproductive stages."),
    ("Farmyard Manure (FYM)", "தொழு உரம்", "Organic matter, N-P-K + micronutrients",
     "All crops", "Basal application before sowing/planting",
     "Improves soil structure, water-holding capacity and microbial activity over the long term."),
    ("Vermicompost", "மண்புழு உரம்", "Organic matter, balanced N-P-K + micronutrients",
     "All crops, especially vegetables", "Basal + top dressing",
     "Slow-release organic fertilizer that improves soil health; widely used in organic farming."),
    ("Single Super Phosphate (SSP)", "சிங்கிள் சூப்பர் பாஸ்பேட்", "Phosphorus (16%) + Calcium + Sulphur",
     "Oilseeds, pulses, general crops", "Basal application at sowing",
     "Economical phosphorus source that also supplies sulphur, useful for oilseed and pulse crops."),
    ("Zinc Sulphate", "துத்தநாக சல்பேட்", "Micronutrient - Zinc",
     "Rice, Citrus, most crops on zinc-deficient soil", "Soil application or foliar spray",
     "Corrects zinc deficiency symptoms such as leaf bronzing/khaira disease in rice; test soil before applying."),
    ("NPK Complex (19:19:19)", "என்பிகே கூட்டு உரம்", "Balanced Nitrogen-Phosphorus-Potassium",
     "Vegetables, fruit crops, general use", "Vegetative and flowering stage, foliar or soil",
     "Water-soluble balanced fertilizer suited for fertigation and foliar feeding programs."),
]
FERT_HEADER = ["fertilizer_id", "fertilizer_name_en", "fertilizer_name_ta", "nutrient_content",
               "suitable_crops", "application_stage", "notes"]
fert_rows = [(f"F{idx+1:03d}", *x) for idx, x in enumerate(FERTILIZERS)]

# ----------------------------------------------------------- IRRIGATION --
IRRIGATION = [
    ("Rice", "Flood/Continuous submergence or AWD (Alternate Wetting & Drying)",
     "2-5 cm standing water during vegetative stage; AWD saves water without yield loss",
     "Maintaining shallow standing water during flowering is critical; avoid stress at panicle initiation."),
    ("Wheat", "Furrow / Sprinkler", "4-6 irrigations at critical growth stages (CRI, tillering, flowering, grain filling)",
     "Crown root initiation (CRI) stage irrigation is the most critical for yield."),
    ("Cotton", "Drip / Furrow", "Irrigate at 50-60% depletion of available soil moisture",
     "Avoid water stress during flowering and boll development; drip irrigation improves water-use efficiency."),
    ("Sugarcane", "Furrow / Drip", "Frequent irrigation every 7-10 days depending on soil and season",
     "High water requirement crop; drip irrigation with fertigation significantly improves efficiency."),
    ("Tomato", "Drip irrigation recommended", "Light, frequent irrigation; avoid water stress during flowering/fruit set",
     "Irregular watering increases blossom-end rot and fruit cracking."),
    ("Groundnut", "Sprinkler / Furrow", "Critical stages: pegging and pod development",
     "Avoid waterlogging; groundnut is sensitive to excess soil moisture at sowing."),
]
IRRIGATION_HEADER = ["crop", "recommended_method", "schedule_guideline", "notes"]

# ------------------------------------------------------------------ PESTS --
PESTS = [
    ("Brown Planthopper", "பழுப்பு தத்துப்பூச்சி", "Rice", "Sap-sucking insect",
     "Hopper burn: yellowing and drying of plants in circular patches",
     "Avoid excess nitrogen, use resistant varieties, encourage natural predators like spiders and mirid bugs."),
    ("Stem Borer", "தண்டு துளைப்பான்", "Rice, Maize, Sugarcane", "Larval boring insect",
     "Dead heart in vegetative stage, white ear head at reproductive stage",
     "Use pheromone traps, remove and destroy affected tillers, encourage egg-parasitoid release (Trichogramma)."),
    ("Whitefly", "வெள்ளை ஈ", "Cotton, Chilli, Tomato", "Sap-sucking insect & virus vector",
     "Yellowing, sooty mould from honeydew, transmits leaf curl viruses",
     "Use yellow sticky traps, avoid excess nitrogen, encourage natural predators; rotate non-host crops."),
    ("Fall Armyworm", "இலை உண்ணும் புழு", "Maize", "Leaf/whorl feeding caterpillar",
     "Window-pane feeding damage on young leaves, ragged holes on whorl leaves",
     "Scout fields regularly, hand-pick egg masses, encourage natural enemies, use pheromone traps for monitoring."),
    ("Pod Borer", "காய் துளைப்பான்", "Groundnut, Blackgram, Pulses", "Larval boring insect",
     "Circular holes bored into pods/flowers, larval frass visible",
     "Deploy pheromone traps, encourage bird perches, avoid continuous monoculture."),
    ("Thrips", "நுண்பூச்சி", "Chilli, Onion, Groundnut", "Rasping-sucking insect",
     "Silvery streaks on leaves, curling and crinkling of young leaves",
     "Use blue sticky traps, avoid moisture stress, encourage predatory mites and bugs."),
]
PESTS_HEADER = ["pest_id", "pest_name_en", "pest_name_ta", "affected_crops", "pest_type",
                "damage_symptoms", "management"]
pests_rows = [(f"P{idx+1:03d}", *x) for idx, x in enumerate(PESTS)]

# ------------------------------------------------------------------- SOIL --
SOIL = [
    ("Alluvial Soil", "வண்டல் மண்", "Rice, Wheat, Sugarcane, Maize",
     "Fertile, well-drained, rich in potash, deposited by rivers",
     "Most fertile Indian soil type found in river plains and deltas; good for a wide crop range."),
    ("Black Cotton Soil (Regur)", "கருப்பு பருத்தி மண்", "Cotton, Groundnut, Sugarcane, Sorghum",
     "High clay content, high water-retention, cracks when dry",
     "Rich in lime, iron, magnesium; retains moisture well but drains slowly, needs careful water management."),
    ("Red Soil", "செம்மண்", "Groundnut, Millets, Pulses, Tobacco",
     "Low nitrogen and organic matter, well-drained, red due to iron oxide",
     "Common in Tamil Nadu uplands; responds well to organic manure and balanced fertilization."),
    ("Laterite Soil", "லேட்டரைட் மண்", "Cashew, Tea, Coffee, Rubber",
     "Acidic, low fertility, high iron/aluminium content",
     "Found in high rainfall regions; needs liming and organic matter to improve fertility."),
    ("Sandy Loam", "மணல் கலந்த களிமண்", "Groundnut, Vegetables, Chilli",
     "Good drainage, moderate water-holding capacity, easy to till",
     "Warms quickly in spring and is easy to work, but may need more frequent irrigation."),
    ("Clayey Soil", "களிமண்", "Rice, Sugarcane",
     "High water-retention, poor drainage, nutrient-rich",
     "Suitable for standing-water crops like rice; needs good drainage management for other crops."),
]
SOIL_HEADER = ["soil_id", "soil_name_en", "soil_name_ta", "suitable_crops",
               "characteristics", "notes"]
soil_rows = [(f"S{idx+1:03d}", *x) for idx, x in enumerate(SOIL)]

# ------------------------------------------------------------------- FAQ --
FAQ = [
    ("How often should tomato crops be irrigated?",
     "Tomato is best irrigated lightly and frequently, especially via drip irrigation, avoiding "
     "both waterlogging and drought stress, particularly during flowering and fruit set.",
     "Irrigation"),
    ("What is the ideal soil pH for most crops?",
     "Most crops perform best in a soil pH range of about 6.0 to 7.5 (slightly acidic to neutral); "
     "get a soil test done for your specific field before amending pH.",
     "Soil"),
    ("When should groundnut be sown?",
     "Groundnut is typically sown at the start of the Kharif season (June-July) or Rabi season "
     "(post-monsoon), depending on regional rainfall patterns.",
     "Cultivation"),
    ("How can I identify nitrogen deficiency in a crop?",
     "Nitrogen deficiency commonly shows as uniform yellowing (chlorosis) starting from older, "
     "lower leaves, along with stunted and slow overall growth.",
     "Fertilizer"),
    ("What causes yellowing of rice leaves?",
     "Yellowing of rice leaves can result from nitrogen deficiency, zinc deficiency, bacterial "
     "leaf blight, or waterlogging/drainage stress; the growth stage and pattern of yellowing help narrow the cause.",
     "Crop Disease"),
    ("How do I control whitefly organically?",
     "Yellow sticky traps, neem-based sprays, avoiding excess nitrogen, and encouraging natural "
     "predators such as ladybird beetles are common organic approaches to manage whitefly.",
     "Pest Management"),
    ("What is crop rotation and why is it important?",
     "Crop rotation is growing different crop types in sequence on the same land across seasons; "
     "it helps break pest/disease cycles, balances soil nutrient use, and improves long-term soil health.",
     "Cultivation"),
    ("How does weather affect fertilizer application?",
     "Avoid applying fertilizer immediately before heavy rain since it can wash away nutrients; "
     "dry, low-wind conditions are generally best for foliar sprays.",
     "Weather"),
    ("When is the best time to harvest rice?",
     "Rice is generally ready for harvest when about 80-85% of the grains have turned golden "
     "yellow and the moisture content of the grain has dropped appropriately.",
     "Harvesting"),
    ("What is Integrated Pest Management (IPM)?",
     "IPM combines biological, cultural, mechanical and (as a last resort) chemical methods to "
     "manage pests sustainably while minimizing environmental and health risks.",
     "Pest Management"),
]
FAQ_HEADER = ["question", "answer", "category"]

# ------------------------------------------------------- TRANSLATIONS --
# Small bilingual glossary used by the fallback dictionary-based translator
# (see src/translation/translator.py) when no internet-connected neural MT
# model is available.
TRANSLATIONS = [
    ("rice", "நெல்"), ("wheat", "கோதுமை"), ("maize", "சோளம்"), ("cotton", "பருத்தி"),
    ("sugarcane", "கரும்பு"), ("groundnut", "நிலக்கடலை"), ("tomato", "தக்காளி"),
    ("banana", "வாழை"), ("chilli", "மிளகாய்"), ("blackgram", "உளுந்து"),
    ("leaves", "இலைகள்"), ("leaf", "இலை"), ("yellow", "மஞ்சள்"), ("yellowing", "மஞ்சளாகிறது"),
    ("crop", "பயிர்"), ("field", "வயல்"), ("water", "தண்ணீர்"), ("soil", "மண்"),
    ("fertilizer", "உரம்"), ("disease", "நோய்"), ("pest", "பூச்சி"), ("irrigation", "நீர்ப்பாசனம்"),
    ("what should i do", "என்ன செய்ய வேண்டும்"), ("what to do", "என்ன செய்ய வேண்டும்"),
    ("my", "என்"), ("is", "ஆகிறது"), ("turning", "ஆகிறது"), ("stage", "நிலை"),
    ("flower", "பூ"), ("flowering", "பூக்கும்"), ("growth", "வளர்ச்சி"),
    ("weather", "வானிலை"), ("rain", "மழை"), ("drought", "வறட்சி"),
    ("symptom", "அறிகுறி"), ("symptoms", "அறிகுறிகள்"), ("wilting", "வாடுதல்"),
    ("root", "வேர்"), ("stem", "தண்டு"), ("fruit", "பழம்"), ("seed", "விதை"),
    ("nitrogen", "நைட்ரஜன்"), ("potassium", "பொட்டாசியம்"), ("phosphorus", "பாஸ்பரஸ்"),
    ("harvest", "அறுவடை"), ("sowing", "விதைப்பு"), ("cultivation", "பயிரிடல்"),
    ("thank you", "நன்றி"), ("hello", "வணக்கம்"), ("help", "உதவி"),
]
TRANS_HEADER = ["term_en", "term_ta"]


def main():
    print("Generating AgroVox agricultural knowledge base ...")
    write_csv(config.CROPS_CSV, CROPS_HEADER, crops_rows)
    write_csv(config.DISEASES_CSV, DISEASES_HEADER, diseases_rows)
    write_csv(config.FERTILIZERS_CSV, FERT_HEADER, fert_rows)
    write_csv(config.IRRIGATION_CSV, IRRIGATION_HEADER, IRRIGATION)
    write_csv(config.PESTS_CSV, PESTS_HEADER, pests_rows)
    write_csv(config.SOIL_CSV, SOIL_HEADER, soil_rows)
    write_csv(config.FAQ_CSV, FAQ_HEADER, FAQ)
    write_csv(config.TRANSLATIONS_CSV, TRANS_HEADER, TRANSLATIONS)
    print("Knowledge base generation complete.")


if __name__ == "__main__":
    main()
