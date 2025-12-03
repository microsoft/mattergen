# Packages
pip install uv
uv venv .venv --python 3.10 
source .venv/bin/activate
uv pip install -e .

# Git LFS
git lfs --version
sudo apt install git-lfs
git lfs install

# Data
git lfs pull -I data-release/mp-20/ --exclude=""
unzip data-release/mp-20/mp_20.zip -d datasets
...
- to add topology labels
python new_work/fine_tuning/add_property.py
...
csv-to-dataset --csv-folder datasets/mp_20/ --dataset-name mp_20 --cache-folder datasets/cache

# Fine-tuning
python new_work/fine_tuning/tune.py

# Generation
python new_work/fine_tuning/gen.py

# Evaluation
git lfs pull -I data-release/alex-mp/reference_MP2020correction.gz --exclude=""
python new_work/evaluations/eval.py

# Generate embeddings for predictor
- uncomment desired embeddings to be generated
python new_work/predictive_model_code/generate_embeddings.py

# Train predictor
python new_work/predictive_model_code/training.py

# Run inference with predictor
python new_work/predictive_model_code/predict.py