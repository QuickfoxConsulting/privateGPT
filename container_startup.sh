#!/bin/bash
  set -e
  
  if [ -n "$HF_TOKEN" ]; then
  echo "Logging into Hugging Face..."
  python -c "from huggingface_hub import login; login(token=\"$HF_TOKEN\")"
  echo "Hugging Face login successful"
  else
  echo "No Hugging Face token provided, skipping login"
  fi
  
  echo "Applying database migrations"
  alembic upgrade head
  
  echo "Loading fixtures"
  poetry run python scripts/fixtures.py
  
  if [ -n "$TIKTOKEN_CACHE_DIR" ]; then
  mkdir -p "$TIKTOKEN_CACHE_DIR"
  TIKTOKEN_URL="https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
  CACHE_FILE="$TIKTOKEN_CACHE_DIR/9b5ad71b4cead37050684f88e"
  if [ ! -f "$CACHE_FILE" ]; then
  echo "Pre-downloading tiktoken encoding..."
  curl -L --retry 5 "$TIKTOKEN_URL" -o "$CACHE_FILE" || echo "Warning: Failed to pre-download tiktoken"
  fi
  fi
  
  echo \"Ensuring fastembed sparse model is present...\"
  mkdir -p /app/models/cache/fastembed
  huggingface-cli download prithivida/Splade_PP_en_v1 --cache-dir /app/models/cache/fastembed || echo \"Warning: Failed to pre-download fastembed model\"
  
  echo "Starting FastAPI server with profile: gemini"
  PYTHONUNBUFFERED=1 PGPT_PROFILES=gemini poetry run python -m uvicorn private_gpt.main:app --host 0.0.0.0 --port 8000 --workers 4
