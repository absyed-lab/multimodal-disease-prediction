# Multimodal Disease Prediction from Symptom Text and Structured Data

A PyTorch prototype that predicts one of 10 common diseases from two inputs together: a patient's free-text symptom description and structured data (demographics, vital signs, history). The two inputs are combined with bidirectional cross-modal attention.

> **Data note:** the code currently trains and evaluates on **synthetic patients** created by `data_loader.py` (template-based symptom text and sampled vitals). Results on this data show that the pipeline works; they are not evidence of clinical performance. Moving to a real dataset such as MIMIC is the next step.

## Model

- **Text encoder:** BioClinicalBERT (`emilyalsentzer/Bio_ClinicalBERT`), projected to 256 dimensions.
- **Structured encoder:** a small TabNet-inspired network over 11 features: age, gender, systolic and diastolic BP, heart rate, temperature, BMI, history of diabetes, hypertension and heart disease, and smoking.
- **Cross-modal attention:** text features attend to structured features and the other way round, and the two enhanced representations are concatenated for classification.
- **Uncertainty:** Monte Carlo dropout (20 samples) gives a confidence score. Predictions below 0.8 confidence are flagged as "refer to a doctor".
- **Interpretability:** attention heatmaps for the text and SHAP values for the structured features.

Three baselines use the same training setup: text only, structured only, and simple concatenation without attention.

## Repository layout

```
multimodal_model.py    model, cross-modal attention and the three baselines
data_loader.py         synthetic data generator and PyTorch dataset
train.py               trains all four models
interpretability.py    attention and SHAP analysis
demo.py                inference and explanation demo
PROJECT_GUIDE.md       longer walkthrough of the project
```

## Run it

```bash
pip install -r requirements.txt
python data_loader.py     # 5,000 synthetic patients, 70/15/15 split
python train.py           # trains the multimodal model and the 3 baselines
python demo.py            # predictions with attention-based explanations
```

Training uses batch size 16, 15 epochs, Adam (lr 1e-4) and ReduceLROnPlateau.

The first run downloads BioClinicalBERT from Hugging Face, so you need internet access once. A GPU is recommended for training.

## Limitations

- Synthetic data only. The symptom text comes from templates, so the task is much easier than real clinical notes.
- 10 disease classes and 11 structured features.
- No validation with clinicians.

## Next steps

- Train and evaluate on MIMIC clinical notes and structured EHR data.
- Check whether the attention weights agree with what clinicians consider important.
- Calibrate the referral threshold on real data.

## Disclaimer

Research prototype, not a medical device, and not for diagnosis.

## License

MIT

## Author

Syed Abdul Basit, M.Sc. AI Engineering, Okan University, Istanbul. Before my M.Sc. I worked for 4.5 years on clinical ML and data pipelines at Aga Khan University Hospital, Karachi.
