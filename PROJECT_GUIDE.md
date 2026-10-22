# Project guide

A longer walkthrough of how to run the code and what each step produces. For the model description and limitations, see the [README](README.md).

## 1. Install

```bash
pip install -r requirements.txt
```

The first run downloads BioClinicalBERT from Hugging Face. A GPU is recommended for training.

## 2. Generate data and train

```bash
python data_loader.py   # writes 5,000 synthetic patients to data/ (70/15/15 split)
python train.py         # trains the multimodal model and the 3 baselines
```

`train.py` saves the best checkpoint of each model to `models/` and a comparison plot to `results/model_comparison.png`.

## 3. Try the model

```bash
python demo.py --examples      # predictions on built-in example cases
python demo.py --interactive   # type your own symptoms and vitals
python demo.py --evaluate      # evaluate on the test split
```

The demo shows the predicted disease, the Monte Carlo dropout confidence, and attention-based explanations.

## Results

No results are published here yet. Anything you get from these scripts comes from synthetic, template-based data, so it shows that the pipeline works, not how the model would do on real patients.

## Troubleshooting

- **Out of memory:** lower `BATCH_SIZE` in `train.py` (default 16).
- **Training too slow on CPU:** use a GPU (for example Google Colab) or lower `EPOCHS` (default 15).
- **Real data:** MIMIC-III is available from PhysioNet after credentialing. Moving to it is the planned next step.
