## Packages
```bash
pip install uv
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -e .
```

## Git LFS
```bash
git lfs --version
sudo apt install git-lfs
git lfs install
```

## Data
```bash
git lfs pull -I data-release/mp-20/ --exclude=""
unzip data-release/mp-20/mp_20.zip -d datasets
```

to add topology labels
```bash
python new_work/fine_tuning/add_property.py
```

```bash
csv-to-dataset --csv-folder datasets/mp_20/ --dataset-name mp_20 --cache-folder datasets/cache
```

## Fine-tuning

```bash
python new_work/fine_tuning/tune.py
```

## Generation
```bash
python new_work/fine_tuning/gen.py
```

## Evaluation
```bash
git lfs pull -I data-release/alex-mp/reference_MP2020correction.gz --exclude=""
python new_work/evaluations/eval.py
```

## Generate embeddings for predictor
uncomment desired embeddings to be generated
```bash
python new_work/predictive_model_code/generate_embeddings.py
```

## Train predictor
```bash
python new_work/predictive_model_code/training.py
```

## Run inference with predictor
```bash
python new_work/predictive_model_code/predict.py