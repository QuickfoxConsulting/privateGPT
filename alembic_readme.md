## **Alembic migrations:**

`alembic init alembic` # initialize the alembic

`alembic revision --autogenerate -m "Create user model"` # first migration

`alembic upgrade 66b63a` # reflect migration on database (here 66b63a) is ssh value


## Local installation
`poetry install --extras "ui llms-llama-cpp embeddings-huggingface vector-stores-qdrant rerank-sentence-transformers vector-stores-postgres storage-nodestore-postgres"`


`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`


 $env:CMAKE_ARGS="-DGGML_CUDA=on"; $env:FORCE_CMAKE=1; pip install --force-reinstall --no-cache-dir llama-cpp-python numpy==1.26.0 



pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir --verbose --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124 numpy==1.26.0 