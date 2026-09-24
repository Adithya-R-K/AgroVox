# Notebooks

This folder is a place to add exploratory Jupyter notebooks for the lab
write-up (`data_analysis.ipynb`, `classification.ipynb`, `ner_training.ipynb`).

The actual, runnable training/evaluation logic lives in `evaluation/*.py`
and `src/*`, so the Streamlit app and `train_classifier.py` do not depend
on notebooks — treat these as optional exploratory companions you can build
on top of the existing modules, e.g.:

```python
import sys; sys.path.insert(0, "..")
from src.classification.classifier import load_dataset, build_features, train_and_compare
df = load_dataset()
result = train_and_compare()
result["comparison"]
```
