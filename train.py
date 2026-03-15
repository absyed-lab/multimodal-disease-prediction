"""
Training Script for Multimodal Disease Prediction
=================================================

Trains and compares:
1. Multimodal Model (Our approach)
2. Text-Only Baseline
3. Structured-Only Baseline
4. Simple Concatenation Baseline

Author: Your Name
Date: March 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

from multimodal_model import (
    MultiModalDiseasePredictor,
    TextOnlyBaseline,
    StructuredOnlyBaseline,
    SimpleConcatBaseline
)
from data_loader import create_data_loaders, DISEASE_CATEGORIES

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =====================================
# TRAINING FUNCTIONS
# =====================================

def train_epoch(model, train_loader, criterion, optimizer, device, model_type='multimodal'):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(train_loader, desc='Training')
    
    for batch in pbar:
        text_inputs = {
            'input_ids': batch['text_inputs']['input_ids'].to(device),
            'attention_mask': batch['text_inputs']['attention_mask'].to(device)
        }
        structured_features = batch['structured_features'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass (different for each model type)
        optimizer.zero_grad()
        
        if model_type == 'text_only':
            outputs = model(text_inputs)
        elif model_type == 'structured_only':
            outputs = model(structured_data=structured_features)
        else:  # multimodal or concat
            outputs = model(text_inputs, structured_features)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device, model_type='multimodal'):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            text_inputs = {
                'input_ids': batch['text_inputs']['input_ids'].to(device),
                'attention_mask': batch['text_inputs']['attention_mask'].to(device)
            }
            structured_features = batch['structured_features'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            if model_type == 'text_only':
                outputs = model(text_inputs)
            elif model_type == 'structured_only':
                outputs = model(structured_data=structured_features)
            else:
                outputs = model(text_inputs, structured_features)
            
            loss = criterion(outputs, labels)
            
            # Track metrics
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Calculate per-class metrics
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
    
    return metrics


# =====================================
# MAIN TRAINING LOOP
# =====================================

def train_model(model, train_loader, val_loader, model_name, model_type, 
                epochs=20, lr=1e-4, save_dir='models'):
    """
    Complete training pipeline for a model
    """
    
    print(f"\n{'='*70}")
    print(f"Training: {model_name}")
    print(f"{'='*70}")
    
    # Setup
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, model_type
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, model_type)
        
        # Update scheduler
        scheduler.step(val_metrics['accuracy'])
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])
        
        # Print metrics
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | Val F1: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), f'{save_dir}/{model_name}_best.pth')
            print(f"✓ Best model saved! Val Acc: {best_val_acc:.4f}")
    
    print(f"\n{'='*70}")
    print(f"Training Complete: {model_name}")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"{'='*70}")
    
    return history, best_val_acc


# =====================================
# COMPREHENSIVE EVALUATION
# =====================================

def comprehensive_evaluation(model, test_loader, model_name, model_type, device):
    """
    Detailed evaluation with uncertainty quantification
    """
    print(f"\n{'='*70}")
    print(f"Comprehensive Evaluation: {model_name}")
    print(f"{'='*70}")
    
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_uncertainties = []
    all_confidences = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluating'):
            text_inputs = {
                'input_ids': batch['text_inputs']['input_ids'].to(device),
                'attention_mask': batch['text_inputs']['attention_mask'].to(device)
            }
            structured_features = batch['structured_features'].to(device)
            labels = batch['label'].to(device)
            
            # Get predictions with uncertainty (if multimodal)
            if model_type == 'multimodal' and hasattr(model, 'predict_with_uncertainty'):
                probs, uncertainty, confidence = model.predict_with_uncertainty(
                    text_inputs, structured_features, num_samples=20
                )
                all_uncertainties.extend(uncertainty.max(dim=1)[0].cpu().numpy())
                all_confidences.extend(confidence.cpu().numpy())
            else:
                # Regular prediction
                if model_type == 'text_only':
                    outputs = model(text_inputs)
                elif model_type == 'structured_only':
                    outputs = model(structured_data=structured_features)
                else:
                    outputs = model(text_inputs, structured_features)
                
                probs = torch.softmax(outputs, dim=1)
                all_confidences.extend(probs.max(dim=1)[0].cpu().numpy())
            
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Per-class metrics
    per_class_acc = []
    for i in range(len(DISEASE_CATEGORIES)):
        mask = np.array(all_labels) == i
        if mask.sum() > 0:
            class_acc = accuracy_score(
                np.array(all_labels)[mask],
                np.array(all_preds)[mask]
            )
            per_class_acc.append(class_acc)
        else:
            per_class_acc.append(0.0)
    
    # Safety metric: % of low-confidence predictions
    low_confidence_threshold = 0.8
    low_confidence_ratio = (np.array(all_confidences) < low_confidence_threshold).mean()
    
    results = {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'per_class_accuracy': per_class_acc,
        'mean_confidence': np.mean(all_confidences),
        'low_confidence_ratio': low_confidence_ratio
    }
    
    # Print results
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    
    print(f"\nPer-Class Accuracy:")
    for disease, acc in zip(DISEASE_CATEGORIES, per_class_acc):
        print(f"  {disease:<30} {acc:.4f}")
    
    print(f"\nSafety Metrics:")
    print(f"  Mean Confidence: {np.mean(all_confidences):.4f}")
    print(f"  Low Confidence Ratio (<0.8): {low_confidence_ratio:.2%}")
    
    if all_uncertainties:
        print(f"  Mean Uncertainty: {np.mean(all_uncertainties):.4f}")
    
    print(f"{'='*70}")
    
    return results


# =====================================
# VISUALIZATION
# =====================================

def plot_comparison(results_dict, save_path='results/model_comparison.png'):
    """
    Plot comparison of all models
    """
    models = list(results_dict.keys())
    metrics = ['accuracy', 'f1', 'precision', 'recall']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, metric in enumerate(metrics):
        values = [results_dict[model][metric] for model in models]
        
        bars = axes[idx].bar(models, values, alpha=0.7, 
                            color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[idx].set_ylabel(metric.capitalize(), fontweight='bold', fontsize=12)
        axes[idx].set_title(f'{metric.capitalize()} Comparison', 
                          fontweight='bold', fontsize=14)
        axes[idx].set_ylim([0.5, 1.0])
        axes[idx].grid(axis='y', alpha=0.3)
        axes[idx].set_xticklabels(models, rotation=15, ha='right')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{val:.3f}', ha='center', va='bottom', 
                          fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to '{save_path}'")
    plt.close()


# =====================================
# MAIN EXECUTION
# =====================================

def main():
    """
    Main training pipeline
    """
    
    print("="*70)
    print("MULTIMODAL DISEASE PREDICTION - TRAINING PIPELINE")
    print("="*70)
    
    # Hyperparameters
    BATCH_SIZE = 16
    EPOCHS = 15
    LEARNING_RATE = 1e-4
    NUM_DISEASES = len(DISEASE_CATEGORIES)
    STRUCTURED_INPUT_DIM = 11  # Number of structured features
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=BATCH_SIZE, num_workers=0
    )
    
    # Define all models
    models_to_train = {
        'Multimodal (Ours)': {
            'model': MultiModalDiseasePredictor(NUM_DISEASES, STRUCTURED_INPUT_DIM),
            'type': 'multimodal'
        },
        'Text-Only': {
            'model': TextOnlyBaseline(NUM_DISEASES),
            'type': 'text_only'
        },
        'Structured-Only': {
            'model': StructuredOnlyBaseline(NUM_DISEASES, STRUCTURED_INPUT_DIM),
            'type': 'structured_only'
        },
        'Simple Concat': {
            'model': SimpleConcatBaseline(NUM_DISEASES, STRUCTURED_INPUT_DIM),
            'type': 'concat'
        }
    }
    
    # Train all models
    training_results = {}
    evaluation_results = {}
    
    for model_name, model_info in models_to_train.items():
        # Train
        history, best_val_acc = train_model(
            model_info['model'],
            train_loader,
            val_loader,
            model_name.replace(' ', '_').replace('(', '').replace(')', ''),
            model_info['type'],
            epochs=EPOCHS,
            lr=LEARNING_RATE
        )
        
        training_results[model_name] = {
            'history': history,
            'best_val_acc': best_val_acc
        }
        
        # Load best model and evaluate
        model_info['model'].load_state_dict(
            torch.load(f'models/{model_name.replace(" ", "_").replace("(", "").replace(")", "")}_best.pth')
        )
        
        eval_results = comprehensive_evaluation(
            model_info['model'],
            test_loader,
            model_name,
            model_info['type'],
            device
        )
        
        evaluation_results[model_name] = eval_results
    
    # Plot comparison
    print("\nGenerating comparison plots...")
    plot_comparison(evaluation_results)
    
    # Save results
    with open('results/training_results.json', 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_results = {}
        for model_name, results in evaluation_results.items():
            json_results[model_name] = {
                'accuracy': float(results['accuracy']),
                'f1': float(results['f1']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'mean_confidence': float(results['mean_confidence'])
            }
        json.dump(json_results, f, indent=2)
    
    print("\n✓ Results saved to 'results/training_results.json'")
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    
    for model_name, results in evaluation_results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  F1-Score:  {results['f1']:.4f}")
    
    # Highlight best model
    best_model = max(evaluation_results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\n🏆 Best Model: {best_model[0]}")
    print(f"   Accuracy: {best_model[1]['accuracy']:.4f}")
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()
