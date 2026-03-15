"""
Demo Script - Multimodal Disease Prediction
===========================================

Interactive demo for:
1. Making predictions on new cases
2. Generating interpretability visualizations
3. Comparing all models
4. Creating publication-ready figures

Author: Your Name
Date: March 2026
"""

import torch
from transformers import AutoTokenizer
import numpy as np
import argparse
import os

from multimodal_model import MultiModalDiseasePredictor
from data_loader import DISEASE_CATEGORIES, create_data_loaders
from interpretability import generate_explanation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =====================================
# LOAD MODEL
# =====================================

def load_trained_model(model_path='models/Multimodal_Ours_best.pth'):
    """Load trained multimodal model"""
    
    NUM_DISEASES = len(DISEASE_CATEGORIES)
    STRUCTURED_INPUT_DIM = 11
    
    model = MultiModalDiseasePredictor(NUM_DISEASES, STRUCTURED_INPUT_DIM)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"✓ Model loaded from '{model_path}'")
        return model
    else:
        print(f"❌ Model not found at '{model_path}'")
        print("Please train model first: python train.py")
        return None


# =====================================
# PREDICTION FUNCTION
# =====================================

def predict_disease(symptom_text, structured_features_dict, model=None, tokenizer=None):
    """
    Make prediction on a new case
    
    Args:
        symptom_text: Patient's symptom description (str)
        structured_features_dict: Dictionary with patient data
            {
                'age': int,
                'gender': 0 or 1,
                'systolic_bp': float,
                'diastolic_bp': float,
                'heart_rate': float,
                'temperature': float,
                'bmi': float,
                'history_diabetes': 0 or 1,
                'history_hypertension': 0 or 1,
                'history_heart_disease': 0 or 1,
                'smoker': 0 or 1
            }
    
    Returns:
        prediction: Disease name
        confidence: Prediction confidence (0-1)
        probabilities: Probability for each disease
    """
    
    if model is None:
        model = load_trained_model()
    
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    
    # Prepare inputs
    encoding = tokenizer(
        symptom_text,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    text_inputs = {
        'input_ids': encoding['input_ids'].to(device),
        'attention_mask': encoding['attention_mask'].to(device)
    }
    
    # Convert structured features to tensor
    feature_order = [
        'age', 'gender', 'systolic_bp', 'diastolic_bp', 'heart_rate',
        'temperature', 'bmi', 'history_diabetes', 'history_hypertension',
        'history_heart_disease', 'smoker'
    ]
    
    structured_array = np.array([structured_features_dict[k] for k in feature_order])
    structured_tensor = torch.tensor(structured_array, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Get prediction with uncertainty
    model.eval()
    with torch.no_grad():
        predictions, uncertainty, confidence = model.predict_with_uncertainty(
            text_inputs, structured_tensor, num_samples=20
        )
    
    # Get top prediction
    pred_idx = predictions.argmax(dim=1).item()
    pred_disease = DISEASE_CATEGORIES[pred_idx]
    pred_confidence = predictions[0, pred_idx].item()
    overall_confidence = confidence[0].item()
    
    # Get all probabilities
    all_probs = predictions[0].cpu().numpy()
    
    return pred_disease, pred_confidence, overall_confidence, all_probs


# =====================================
# INTERACTIVE DEMO
# =====================================

def interactive_demo():
    """Interactive command-line demo"""
    
    print("="*70)
    print("MULTIMODAL DISEASE PREDICTION - INTERACTIVE DEMO")
    print("="*70)
    
    # Load model
    model = load_trained_model()
    if model is None:
        return
    
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    
    print("\nEnter patient information:")
    print("-" * 70)
    
    # Get symptom text
    symptom_text = input("Symptom Description: ")
    
    # Get structured features
    print("\nStructured Features:")
    age = int(input("  Age: "))
    gender = int(input("  Gender (0=Female, 1=Male): "))
    systolic_bp = float(input("  Systolic BP: "))
    diastolic_bp = float(input("  Diastolic BP: "))
    heart_rate = float(input("  Heart Rate: "))
    temperature = float(input("  Temperature (°C): "))
    bmi = float(input("  BMI: "))
    history_diabetes = int(input("  History of Diabetes (0=No, 1=Yes): "))
    history_hypertension = int(input("  History of Hypertension (0=No, 1=Yes): "))
    history_heart_disease = int(input("  History of Heart Disease (0=No, 1=Yes): "))
    smoker = int(input("  Smoker (0=No, 1=Yes): "))
    
    structured_features = {
        'age': age,
        'gender': gender,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'heart_rate': heart_rate,
        'temperature': temperature,
        'bmi': bmi,
        'history_diabetes': history_diabetes,
        'history_hypertension': history_hypertension,
        'history_heart_disease': history_heart_disease,
        'smoker': smoker
    }
    
    # Make prediction
    print("\nPredicting...")
    pred_disease, pred_conf, overall_conf, all_probs = predict_disease(
        symptom_text, structured_features, model, tokenizer
    )
    
    # Display results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    print(f"\nPredicted Disease: {pred_disease}")
    print(f"Prediction Confidence: {pred_conf:.1%}")
    print(f"Overall Confidence Score: {overall_conf:.1%}")
    
    if overall_conf < 0.8:
        print("\n⚠️  WARNING: LOW CONFIDENCE")
        print("The model is uncertain about this prediction.")
        print("RECOMMENDATION: Consult a healthcare professional.")
    else:
        print("\n✓ High confidence prediction")
        print("Note: Always consult a healthcare professional for diagnosis.")
    
    # Show top 3 predictions
    print("\nTop 3 Predictions:")
    top3_idx = np.argsort(all_probs)[-3:][::-1]
    for i, idx in enumerate(top3_idx, 1):
        print(f"  {i}. {DISEASE_CATEGORIES[idx]:<30} {all_probs[idx]:.1%}")
    
    print("\n" + "="*70)
    
    # Ask about interpretability
    generate_interpret = input("\nGenerate interpretability visualizations? (y/n): ")
    if generate_interpret.lower() == 'y':
        # For demo, we assume true_label = prediction (in practice, you'd know the true label)
        true_label = pred_disease
        true_label_idx = DISEASE_CATEGORIES.index(true_label)
        
        # Normalize structured features (simple standardization for demo)
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        structured_array = np.array([structured_features[k] for k in [
            'age', 'gender', 'systolic_bp', 'diastolic_bp', 'heart_rate',
            'temperature', 'bmi', 'history_diabetes', 'history_hypertension',
            'history_heart_disease', 'smoker'
        ]])
        structured_normalized = scaler.fit_transform(structured_array.reshape(1, -1))[0]
        
        generate_explanation(
            model, tokenizer, device,
            symptom_text, structured_normalized, true_label_idx,
            save_dir='results/demo_interpretation'
        )


# =====================================
# EXAMPLE CASES
# =====================================

def run_example_cases():
    """Run predictions on example cases"""
    
    print("="*70)
    print("EXAMPLE CASES")
    print("="*70)
    
    examples = [
        {
            'name': 'Case 1: Diabetes',
            'symptom': "I've been feeling very thirsty and urinating frequently, also experiencing fatigue",
            'structured': {
                'age': 55,
                'gender': 1,
                'systolic_bp': 135,
                'diastolic_bp': 88,
                'heart_rate': 78,
                'temperature': 36.8,
                'bmi': 32,
                'history_diabetes': 0,
                'history_hypertension': 1,
                'history_heart_disease': 0,
                'smoker': 0
            }
        },
        {
            'name': 'Case 2: Pneumonia',
            'symptom': "I have high fever, productive cough with phlegm, and difficulty breathing",
            'structured': {
                'age': 42,
                'gender': 0,
                'systolic_bp': 118,
                'diastolic_bp': 76,
                'heart_rate': 95,
                'temperature': 38.9,
                'bmi': 24,
                'history_diabetes': 0,
                'history_hypertension': 0,
                'history_heart_disease': 0,
                'smoker': 1
            }
        },
        {
            'name': 'Case 3: Hypertension',
            'symptom': "I have severe headaches and feel dizzy, also experiencing shortness of breath",
            'structured': {
                'age': 62,
                'gender': 1,
                'systolic_bp': 165,
                'diastolic_bp': 102,
                'heart_rate': 88,
                'temperature': 37.1,
                'bmi': 29,
                'history_diabetes': 1,
                'history_hypertension': 0,
                'history_heart_disease': 1,
                'smoker': 0
            }
        }
    ]
    
    model = load_trained_model()
    if model is None:
        return
    
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    
    for example in examples:
        print(f"\n{example['name']}")
        print("-" * 70)
        print(f"Symptoms: {example['symptom']}")
        print(f"Age: {example['structured']['age']}, Gender: {'Male' if example['structured']['gender'] else 'Female'}")
        
        pred_disease, pred_conf, overall_conf, _ = predict_disease(
            example['symptom'], example['structured'], model, tokenizer
        )
        
        print(f"\n→ Prediction: {pred_disease} (Confidence: {pred_conf:.1%})")
        
        if overall_conf < 0.8:
            print("⚠️  Low confidence - refer to doctor")
        else:
            print("✓ High confidence")
    
    print("\n" + "="*70)


# =====================================
# MAIN
# =====================================

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(description='Multimodal Disease Prediction Demo')
    parser.add_argument('--interactive', action='store_true', help='Run interactive demo')
    parser.add_argument('--examples', action='store_true', help='Run example cases')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate on test set')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_demo()
    elif args.examples:
        run_example_cases()
    elif args.evaluate:
        print("Running full evaluation...")
        print("Use: python train.py (evaluation included)")
    else:
        # Default: show examples
        run_example_cases()
        print("\nFor interactive demo, run: python demo.py --interactive")


if __name__ == '__main__':
    main()
