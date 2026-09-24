"""
AgroVox - Agricultural Knowledge Base access layer.

Thin pandas-backed accessor over the structured CSV knowledge base
(crops, diseases, fertilizers, irrigation, pests, soil, FAQ) used by the
Knowledge Explorer UI page and by the QA engine's answer formatting.
"""
import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config


class KnowledgeBase:
    def __init__(self):
        self.crops = pd.read_csv(config.CROPS_CSV)
        self.diseases = pd.read_csv(config.DISEASES_CSV)
        self.fertilizers = pd.read_csv(config.FERTILIZERS_CSV)
        self.irrigation = pd.read_csv(config.IRRIGATION_CSV)
        self.pests = pd.read_csv(config.PESTS_CSV)
        self.soil = pd.read_csv(config.SOIL_CSV)
        self.faq = pd.read_csv(config.FAQ_CSV)

    def search(self, table: str, query: str) -> pd.DataFrame:
        df = getattr(self, table)
        if not query:
            return df
        query = query.lower()
        mask = df.apply(lambda row: row.astype(str).str.lower().str.contains(query, na=False).any(), axis=1)
        return df[mask]

    def get_tables(self):
        return {
            "Crops": self.crops, "Diseases": self.diseases, "Fertilizers": self.fertilizers,
            "Irrigation": self.irrigation, "Pests": self.pests, "Soil": self.soil,
            "FAQ": self.faq,
        }

    def stats(self) -> dict:
        return {name: len(df) for name, df in self.get_tables().items()}


if __name__ == "__main__":
    kb = KnowledgeBase()
    print(kb.stats())
    print(kb.search("diseases", "rice"))
