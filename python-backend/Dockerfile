# AgroVox - Production Dockerfile
# Multi-stage-free, lightweight CPU-only image (no GPU/CUDA needed for the
# default local fallback pipeline - see README.md -> Fallback Strategy).

FROM python:3.11-slim

LABEL maintainer="AgroVox Project"
LABEL description="Voice-Driven Intelligence for Smarter Farming - bilingual agricultural NLP assistant"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps needed by scikit-learn/scipy/matplotlib wheels and healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

# Pre-download the NLTK corpora needed by preprocessing/BLEU at build time
# so the container has no first-request latency or hidden network calls.
RUN python -c "import nltk; [nltk.download(p) for p in ['punkt','punkt_tab','stopwords','wordnet','omw-1.4']]"

COPY . .

# Generate datasets and train/evaluate all models at image build time, so
# the running container starts instantly with models already on disk.
RUN python train_classifier.py

# Run the ETL pipeline once + generate the EDA report, so the "Data
# Pipeline (ETL & EDA)" tab has data to show immediately on first load.
RUN python etl/etl_pipeline.py && python eda/exploratory_analysis.py

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
