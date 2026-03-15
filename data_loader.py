"""
Data Loading and Preprocessing
===============================

Handles:
1. MIMIC-III data (when available)
2. Synthea synthetic data generation
3. HealthSearchQA symptom queries
4. Data preprocessing and augmentation

Author: Your Name
Date: March 2026
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import json
import os

# =====================================
# DISEASE CATEGORIES
# =====================================

# 10 common diseases for initial study
DISEASE_CATEGORIES = [
    'Diabetes',
    'Hypertension', 
    'Heart Disease',
    'Asthma',
    'Pneumonia',
    'Urinary Tract Infection',
    'Gastroenteritis',
    'Migraine',
    'Depression',
    'Arthritis'
]

# Symptom templates for each disease
SYMPTOM_TEMPLATES = {
    'Diabetes': [
        "I've been feeling very thirsty and urinating frequently",
        "Experiencing extreme hunger and fatigue lately",
        "I have blurred vision and slow healing wounds",
        "Constant thirst, frequent urination, and unexplained weight loss"
    ],
    'Hypertension': [
        "I have severe headaches and feel dizzy",
        "Experiencing shortness of breath and chest pain",
        "Frequent nosebleeds and severe headaches",
        "Feel anxious with irregular heartbeat"
    ],
    'Heart Disease': [
        "I have chest pain and shortness of breath",
        "Experiencing pain in arms, neck, and jaw",
        "Feel nauseous and extremely fatigued",
        "Chest discomfort and cold sweats"
    ],
    'Asthma': [
        "I have difficulty breathing and wheezing",
        "Experiencing tightness in chest and coughing",
        "Shortness of breath especially at night",
        "Wheezing and coughing, worse with exercise"
    ],
    'Pneumonia': [
        "I have high fever and productive cough",
        "Experiencing chest pain when breathing and fatigue",
        "Fever, cough with phlegm, and difficulty breathing",
        "Severe cough, fever, and shortness of breath"
    ],
    'Urinary Tract Infection': [
        "I have burning sensation when urinating",
        "Experiencing frequent urge to urinate and cloudy urine",
        "Lower abdominal pain and frequent urination",
        "Painful urination and strong-smelling urine"
    ],
    'Gastroenteritis': [
        "I have severe diarrhea and stomach cramps",
        "Experiencing nausea, vomiting, and fever",
        "Watery diarrhea and abdominal pain",
        "Vomiting, diarrhea, and dehydration symptoms"
    ],
    'Migraine': [
        "I have severe throbbing headache on one side",
        "Experiencing sensitivity to light and nausea",
        "Pulsating pain with visual disturbances",
        "Severe headache with nausea and light sensitivity"
    ],
    'Depression': [
        "I feel sad and have lost interest in activities",
        "Experiencing fatigue and difficulty concentrating",
        "Feeling hopeless and having sleep problems",
        "Persistent sadness and loss of energy"
    ],
    'Arthritis': [
        "I have joint pain and stiffness in the morning",
        "Experiencing swelling and reduced range of motion",
        "Joint pain that worsens with activity",
        "Stiff joints and swelling, especially in hands"
    ]
}

# =====================================
# SYNTHETIC DATA GENERATOR
# =====================================

class SyntheticDataGenerator:
    """
    Generate synthetic medical data similar to MIMIC-III/Synthea format
    Used for development and testing until real data is available
    """
    
    def __init__(self, num_samples=5000, seed=42):
        self.num_samples = num_samples
        self.seed = seed
        np.random.seed(seed)
    
    def generate_patient_data(self):
        """
        Generate synthetic patient records with:
        - Text symptoms
        - Structured features (demographics, vitals, history)
        - Disease labels
        """
        
        data = []
        
        for i in range(self.num_samples):
            # Select disease
            disease = np.random.choice(DISEASE_CATEGORIES)
            disease_idx = DISEASE_CATEGORIES.index(disease)
            
            # Generate text symptom description
            symptom_template = np.random.choice(SYMPTOM_TEMPLATES[disease])
            
            # Add some variation
            variations = [
                " for the past few days",
                " which started last week",
                " that's been getting worse",
                ""
            ]
            symptom_text = symptom_template + np.random.choice(variations)
            
            # Generate structured features
            # Demographics
            age = np.random.randint(18, 85)
            gender = np.random.choice([0, 1])  # 0=Female, 1=Male
            
            # Vitals (with disease-specific patterns)
            if disease == 'Hypertension' or disease == 'Heart Disease':
                systolic_bp = np.random.normal(150, 15)
                diastolic_bp = np.random.normal(95, 10)
                heart_rate = np.random.normal(85, 12)
            elif disease == 'Diabetes':
                systolic_bp = np.random.normal(130, 10)
                diastolic_bp = np.random.normal(85, 8)
                heart_rate = np.random.normal(75, 10)
                blood_glucose = np.random.normal(180, 30)  # High
            else:
                systolic_bp = np.random.normal(120, 10)
                diastolic_bp = np.random.normal(80, 8)
                heart_rate = np.random.normal(75, 10)
            
            temperature = np.random.normal(37.0, 0.5) if disease != 'Pneumonia' else np.random.normal(38.5, 0.5)
            
            # Medical history (binary flags)
            history_diabetes = 1 if disease == 'Diabetes' else np.random.choice([0, 1], p=[0.9, 0.1])
            history_hypertension = 1 if disease == 'Hypertension' else np.random.choice([0, 1], p=[0.85, 0.15])
            history_heart_disease = 1 if disease == 'Heart Disease' else np.random.choice([0, 1], p=[0.95, 0.05])
            
            # Lifestyle factors
            smoker = np.random.choice([0, 1], p=[0.7, 0.3])
            bmi = np.random.normal(26, 4)
            
            # Create record
            record = {
                'patient_id': f'P{i:05d}',
                'symptom_text': symptom_text,
                'age': age,
                'gender': gender,
                'systolic_bp': max(90, min(200, systolic_bp)),
                'diastolic_bp': max(60, min(120, diastolic_bp)),
                'heart_rate': max(50, min(120, heart_rate)),
                'temperature': max(35, min(40, temperature)),
                'bmi': max(15, min(45, bmi)),
                'history_diabetes': history_diabetes,
                'history_hypertension': history_hypertension,
                'history_heart_disease': history_heart_disease,
                'smoker': smoker,
                'disease': disease,
                'disease_idx': disease_idx
            }
            
            data.append(record)
        
        return pd.DataFrame(data)
    
    def save_data(self, output_dir='data'):
        """Save generated data to disk"""
        os.makedirs(output_dir, exist_ok=True)
        
        df = self.generate_patient_data()
        
        # Save full dataset
        df.to_csv(f'{output_dir}/synthetic_patient_data.csv', index=False)
        
        # Save train/val/test splits
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=self.seed, stratify=df['disease'])
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=self.seed, stratify=temp_df['disease'])
        
        train_df.to_csv(f'{output_dir}/train.csv', index=False)
        val_df.to_csv(f'{output_dir}/val.csv', index=False)
        test_df.to_csv(f'{output_dir}/test.csv', index=False)
        
        print(f"✓ Generated {len(df)} synthetic patient records")
        print(f"  - Train: {len(train_df)} samples")
        print(f"  - Val: {len(val_df)} samples")
        print(f"  - Test: {len(test_df)} samples")
        print(f"✓ Data saved to '{output_dir}/' directory")
        
        return train_df, val_df, test_df


# =====================================
# PYTORCH DATASET
# =====================================

class DiseaseDataset(Dataset):
    """
    PyTorch Dataset for multimodal disease prediction
    
    Returns:
        text_inputs: Tokenized symptom descriptions
        structured_features: Demographics + vitals + history
        labels: Disease labels
        patient_ids: For tracking
    """
    
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Structured feature columns
        self.structured_features = [
            'age', 'gender', 'systolic_bp', 'diastolic_bp', 'heart_rate',
            'temperature', 'bmi', 'history_diabetes', 'history_hypertension',
            'history_heart_disease', 'smoker'
        ]
        
        # Normalize structured features
        self.scaler = StandardScaler()
        self.df[self.structured_features] = self.scaler.fit_transform(
            self.df[self.structured_features]
        )
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Text input (tokenized)
        symptom_text = row['symptom_text']
        text_encoding = self.tokenizer(
            symptom_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        text_inputs = {
            'input_ids': text_encoding['input_ids'].squeeze(0),
            'attention_mask': text_encoding['attention_mask'].squeeze(0)
        }
        
        # Structured features
        structured_features = torch.tensor(
            row[self.structured_features].values,
            dtype=torch.float32
        )
        
        # Label
        label = torch.tensor(row['disease_idx'], dtype=torch.long)
        
        # Patient ID (for tracking)
        patient_id = row['patient_id']
        
        return {
            'text_inputs': text_inputs,
            'structured_features': structured_features,
            'label': label,
            'patient_id': patient_id
        }


# =====================================
# DATA LOADER CREATION
# =====================================

def create_data_loaders(batch_size=32, num_workers=2):
    """
    Create train, validation, and test data loaders
    
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    
    # Check if data exists, otherwise generate
    data_dir = 'data'
    if not os.path.exists(f'{data_dir}/train.csv'):
        print("Generating synthetic data...")
        generator = SyntheticDataGenerator(num_samples=5000)
        generator.save_data(data_dir)
    
    # Load datasets
    print("\nLoading datasets...")
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    val_df = pd.read_csv(f'{data_dir}/val.csv')
    test_df = pd.read_csv(f'{data_dir}/test.csv')
    
    print(f"✓ Train: {len(train_df)} samples")
    print(f"✓ Val: {len(val_df)} samples")
    print(f"✓ Test: {len(test_df)} samples")
    
    # Create datasets
    train_dataset = DiseaseDataset(train_df, tokenizer)
    val_dataset = DiseaseDataset(val_df, tokenizer)
    test_dataset = DiseaseDataset(test_df, tokenizer)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# =====================================
# DATASET STATISTICS
# =====================================

def print_dataset_statistics(data_dir='data'):
    """Print detailed statistics about the dataset"""
    
    print("\n" + "="*70)
    print("DATASET STATISTICS")
    print("="*70)
    
    # Load data
    train_df = pd.read_csv(f'{data_dir}/train.csv')
    
    # Disease distribution
    print("\nDisease Distribution:")
    disease_counts = train_df['disease'].value_counts()
    for disease, count in disease_counts.items():
        percentage = (count / len(train_df)) * 100
        print(f"  {disease:<30} {count:>5} ({percentage:>5.1f}%)")
    
    # Demographics
    print("\nDemographics:")
    print(f"  Age: {train_df['age'].mean():.1f} ± {train_df['age'].std():.1f} years")
    print(f"  Gender: {(train_df['gender'].sum() / len(train_df) * 100):.1f}% male")
    
    # Vitals
    print("\nVital Signs (mean ± std):")
    print(f"  Blood Pressure: {train_df['systolic_bp'].mean():.0f}/{train_df['diastolic_bp'].mean():.0f} mmHg")
    print(f"  Heart Rate: {train_df['heart_rate'].mean():.0f} ± {train_df['heart_rate'].std():.0f} bpm")
    print(f"  Temperature: {train_df['temperature'].mean():.1f} ± {train_df['temperature'].std():.1f} °C")
    print(f"  BMI: {train_df['bmi'].mean():.1f} ± {train_df['bmi'].std():.1f}")
    
    # Symptom text length
    train_df['text_length'] = train_df['symptom_text'].str.split().str.len()
    print("\nSymptom Text:")
    print(f"  Average length: {train_df['text_length'].mean():.1f} words")
    print(f"  Range: {train_df['text_length'].min()}-{train_df['text_length'].max()} words")
    
    print("="*70)


if __name__ == '__main__':
    """
    Test data loading pipeline
    """
    print("="*70)
    print("DATA LOADING PIPELINE TEST")
    print("="*70)
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    generator = SyntheticDataGenerator(num_samples=5000)
    train_df, val_df, test_df = generator.save_data('data')
    
    # Print statistics
    print_dataset_statistics('data')
    
    # Create data loaders
    print("\n2. Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(batch_size=8)
    print(f"✓ Data loaders created")
    print(f"  Batch size: 8")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Test one batch
    print("\n3. Testing one batch...")
    batch = next(iter(train_loader))
    
    print(f"✓ Batch loaded successfully")
    print(f"  Text input_ids shape: {batch['text_inputs']['input_ids'].shape}")
    print(f"  Text attention_mask shape: {batch['text_inputs']['attention_mask'].shape}")
    print(f"  Structured features shape: {batch['structured_features'].shape}")
    print(f"  Labels shape: {batch['label'].shape}")
    print(f"  Number of patients: {len(batch['patient_id'])}")
    
    # Show sample
    print("\n4. Sample patient record:")
    print(f"  Patient ID: {batch['patient_id'][0]}")
    print(f"  Symptom text (tokenized): {batch['text_inputs']['input_ids'][0][:20]}...")
    print(f"  Structured features: {batch['structured_features'][0]}")
    print(f"  Disease label: {batch['label'][0].item()} ({DISEASE_CATEGORIES[batch['label'][0]]})")
    
    print("\n" + "="*70)
    print("✓ ALL DATA LOADING TESTS PASSED!")
    print("="*70)
    print("\nData is ready for training!")
