"""
Interpretability Module for Multimodal Disease Prediction
=========================================================

Implements:
1. Text Attention Visualization
2. SHAP Values for Structured Features  
3. Cross-Modal Attention Analysis
4. Feature Importance Ranking

Novel Contribution: Clinician-understandable explanations

Author: Your Name
Date: March 2026
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer
import shap
from captum.attr import IntegratedGradients, LayerIntegratedGradients
import pandas as pd
import os

from multimodal_model import MultiModalDiseasePredictor
from data_loader import DISEASE_CATEGORIES

# =====================================
# ATTENTION VISUALIZATION
# =====================================

class AttentionVisualizer:
    """
    Visualize attention weights to show which words/symptoms
    the model focuses on when making predictions
    """
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def visualize_text_attention(self, symptom_text, structured_features, 
                                 true_label, save_path=None):
        """
        Create heatmap showing which words in symptom description
        are most important for the prediction
        """
        
        # Tokenize
        encoding = self.tokenizer(
            symptom_text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        text_inputs = {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }
        
        structured_tensor = torch.tensor(
            structured_features, dtype=torch.float32
        ).unsqueeze(0).to(self.device)
        
        # Get prediction with attention
        self.model.eval()
        with torch.no_grad():
            logits, attention_dict = self.model(
                text_inputs, structured_tensor, return_attention=True
            )
            prediction = torch.argmax(logits, dim=1).item()
            confidence = torch.softmax(logits, dim=1).max().item()
        
        # Extract attention weights
        text_attention = attention_dict['text_attention'][0].cpu().numpy()
        
        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(text_inputs['input_ids'][0])
        
        # Only show non-padding tokens
        valid_length = text_attention.sum()
        tokens = tokens[:int(valid_length)]
        attention_weights = text_attention[:int(valid_length)]
        
        # Create visualization
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Attention heatmap
        attention_matrix = attention_weights.reshape(1, -1)
        sns.heatmap(attention_matrix, 
                   xticklabels=tokens,
                   yticklabels=['Attention'],
                   cmap='YlOrRd',
                   ax=axes[0],
                   cbar_kws={'label': 'Attention Weight'})
        axes[0].set_title('Text Attention Weights', fontweight='bold', fontsize=14)
        axes[0].set_xticklabels(tokens, rotation=45, ha='right')
        
        # Bar chart of top words
        top_k = min(10, len(tokens))
        top_indices = np.argsort(attention_weights)[-top_k:][::-1]
        top_tokens = [tokens[i] for i in top_indices]
        top_weights = attention_weights[top_indices]
        
        axes[1].barh(range(top_k), top_weights, color='coral', alpha=0.7)
        axes[1].set_yticks(range(top_k))
        axes[1].set_yticklabels(top_tokens)
        axes[1].set_xlabel('Attention Weight', fontweight='bold')
        axes[1].set_title(f'Top {top_k} Most Important Words', fontweight='bold', fontsize=14)
        axes[1].invert_yaxis()
        axes[1].grid(axis='x', alpha=0.3)
        
        # Add prediction info
        pred_disease = DISEASE_CATEGORIES[prediction]
        true_disease = DISEASE_CATEGORIES[true_label]
        color = 'green' if prediction == true_label else 'red'
        
        fig.suptitle(
            f'Prediction: {pred_disease} (Confidence: {confidence:.2%})\n'
            f'True Label: {true_disease}',
            fontsize=16, fontweight='bold', color=color, y=0.98
        )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Attention visualization saved to '{save_path}'")
        
        plt.show()
        plt.close()
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'top_words': list(zip(top_tokens, top_weights))
        }


# =====================================
# STRUCTURED FEATURE IMPORTANCE (SHAP)
# =====================================

class SHAPExplainer:
    """
    Use SHAP to explain which structured features
    (age, vitals, history) are important for predictions
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
        # Feature names
        self.feature_names = [
            'Age', 'Gender', 'Systolic BP', 'Diastolic BP', 'Heart Rate',
            'Temperature', 'BMI', 'History: Diabetes', 'History: Hypertension',
            'History: Heart Disease', 'Smoker'
        ]
    
    def analyze_feature_importance(self, structured_data, text_inputs_list, 
                                   num_samples=100, save_path=None):
        """
        Use SHAP to analyze feature importance across multiple samples
        
        Args:
            structured_data: numpy array (num_samples, num_features)
            text_inputs_list: list of tokenized text inputs
        """
        
        # Wrapper function for SHAP
        def predict_fn(structured_batch):
            """Prediction function for SHAP"""
            batch_size = structured_batch.shape[0]
            predictions = []
            
            self.model.eval()
            with torch.no_grad():
                for i in range(batch_size):
                    # Use first text input as representative (SHAP needs fixed text)
                    text_input = text_inputs_list[0]
                    
                    struct_tensor = torch.tensor(
                        structured_batch[i:i+1], dtype=torch.float32
                    ).to(self.device)
                    
                    logits = self.model(text_input, struct_tensor)
                    probs = torch.softmax(logits, dim=1)
                    predictions.append(probs.cpu().numpy())
            
            return np.vstack(predictions)
        
        # Create SHAP explainer
        print("Calculating SHAP values (this may take a few minutes)...")
        explainer = shap.KernelExplainer(predict_fn, structured_data[:100])
        shap_values = explainer.shap_values(structured_data[:num_samples])
        
        # Aggregate SHAP values across all classes
        if isinstance(shap_values, list):
            # Average absolute SHAP values across classes
            shap_values_agg = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            shap_values_agg = np.abs(shap_values)
        
        # Calculate mean importance per feature
        mean_importance = np.mean(shap_values_agg, axis=0)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar plot of feature importance
        sorted_idx = np.argsort(mean_importance)
        axes[0].barh(range(len(self.feature_names)), 
                    mean_importance[sorted_idx],
                    color='steelblue', alpha=0.7)
        axes[0].set_yticks(range(len(self.feature_names)))
        axes[0].set_yticklabels([self.feature_names[i] for i in sorted_idx])
        axes[0].set_xlabel('Mean |SHAP Value|', fontweight='bold', fontsize=12)
        axes[0].set_title('Feature Importance', fontweight='bold', fontsize=14)
        axes[0].grid(axis='x', alpha=0.3)
        
        # Heatmap of SHAP values
        sample_indices = np.random.choice(num_samples, min(20, num_samples), replace=False)
        sns.heatmap(shap_values_agg[sample_indices].T,
                   yticklabels=self.feature_names,
                   cmap='RdBu_r',
                   center=0,
                   ax=axes[1],
                   cbar_kws={'label': '|SHAP Value|'})
        axes[1].set_title('SHAP Values per Sample', fontweight='bold', fontsize=14)
        axes[1].set_xlabel('Sample Index', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ SHAP visualization saved to '{save_path}'")
        
        plt.show()
        plt.close()
        
        return {
            'feature_names': self.feature_names,
            'mean_importance': mean_importance,
            'shap_values': shap_values_agg
        }


# =====================================
# CROSS-MODAL ATTENTION ANALYSIS
# =====================================

class CrossModalAnalyzer:
    """
    Analyze how text symptoms and structured features interact
    via cross-modal attention mechanism
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def visualize_cross_modal_attention(self, text_inputs, structured_features,
                                       symptom_text, save_path=None):
        """
        Visualize cross-modal attention:
        - How text attends to structured features
        - How structured features attend to text
        """
        
        self.model.eval()
        with torch.no_grad():
            logits, attention_dict = self.model(
                text_inputs, structured_features, return_attention=True
            )
        
        # Extract cross-modal attention
        attn_t2s = attention_dict['text_to_structured'][0].cpu().numpy().squeeze()
        attn_s2t = attention_dict['structured_to_text'][0].cpu().numpy().squeeze()
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Text to Structured attention
        axes[0].bar(range(1), [attn_t2s], color='coral', alpha=0.7)
        axes[0].set_ylabel('Attention Weight', fontweight='bold')
        axes[0].set_title('Text → Structured Features Attention', 
                         fontweight='bold', fontsize=14)
        axes[0].set_ylim([0, 1])
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].text(0, attn_t2s + 0.05, f'{attn_t2s:.3f}', 
                    ha='center', fontweight='bold')
        
        # Structured to Text attention
        axes[1].bar(range(1), [attn_s2t], color='steelblue', alpha=0.7)
        axes[1].set_ylabel('Attention Weight', fontweight='bold')
        axes[1].set_title('Structured Features → Text Attention',
                         fontweight='bold', fontsize=14)
        axes[1].set_ylim([0, 1])
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].text(0, attn_s2t + 0.05, f'{attn_s2t:.3f}',
                    ha='center', fontweight='bold')
        
        fig.suptitle('Cross-Modal Attention Analysis', 
                    fontsize=16, fontweight='bold', y=1.00)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Cross-modal attention saved to '{save_path}'")
        
        plt.show()
        plt.close()


# =====================================
# COMPREHENSIVE EXPLANATION
# =====================================

def generate_explanation(model, tokenizer, device, 
                        symptom_text, structured_features, true_label,
                        save_dir='results/interpretability'):
    """
    Generate comprehensive explanation for a single prediction
    
    Outputs:
    1. Text attention visualization
    2. Structured feature importance
    3. Cross-modal attention
    4. Natural language explanation
    """
    
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*70)
    print("GENERATING COMPREHENSIVE EXPLANATION")
    print("="*70)
    
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
    
    struct_tensor = torch.tensor(
        structured_features, dtype=torch.float32
    ).unsqueeze(0).to(device)
    
    # Get prediction with uncertainty
    model.eval()
    predictions, uncertainty, confidence = model.predict_with_uncertainty(
        text_inputs, struct_tensor, num_samples=20
    )
    
    prediction = predictions.argmax(dim=1).item()
    pred_prob = predictions[0, prediction].item()
    uncertainty_val = uncertainty[0, prediction].item()
    confidence_val = confidence[0].item()
    
    # 1. Text Attention
    print("\n1. Analyzing text attention...")
    attn_viz = AttentionVisualizer(model, tokenizer, device)
    attn_results = attn_viz.visualize_text_attention(
        symptom_text, structured_features, true_label,
        save_path=f'{save_dir}/text_attention.png'
    )
    
    # 2. Cross-Modal Attention
    print("\n2. Analyzing cross-modal attention...")
    cross_modal = CrossModalAnalyzer(model, device)
    cross_modal.visualize_cross_modal_attention(
        text_inputs, struct_tensor, symptom_text,
        save_path=f'{save_dir}/cross_modal_attention.png'
    )
    
    # 3. Generate natural language explanation
    print("\n3. Generating explanation...")
    
    explanation = f"""
PATIENT CASE EXPLANATION
=======================

Symptom Description:
"{symptom_text}"

PREDICTION:
-----------
Predicted Disease: {DISEASE_CATEGORIES[prediction]}
Confidence: {pred_prob:.1%}
Uncertainty: {uncertainty_val:.3f}
Overall Confidence Score: {confidence_val:.1%}

SAFETY RECOMMENDATION:
---------------------
"""
    
    if confidence_val < 0.8:
        explanation += "⚠️  LOW CONFIDENCE - REFER TO DOCTOR\n"
        explanation += "The model is uncertain about this prediction. "
        explanation += "Please consult a healthcare professional for accurate diagnosis.\n"
    else:
        explanation += "✓ Model is confident in this prediction.\n"
        explanation += "However, always consult a healthcare professional for final diagnosis.\n"
    
    explanation += f"""

KEY FACTORS IN PREDICTION:
--------------------------
Top Important Symptoms (from text):
"""
    
    for i, (word, weight) in enumerate(attn_results['top_words'][:5], 1):
        explanation += f"  {i}. '{word}' (importance: {weight:.3f})\n"
    
    explanation += "\n"
    
    if prediction == true_label:
        explanation += "✓ CORRECT PREDICTION\n"
    else:
        explanation += f"✗ INCORRECT - True disease was: {DISEASE_CATEGORIES[true_label]}\n"
    
    # Save explanation
    with open(f'{save_dir}/explanation.txt', 'w') as f:
        f.write(explanation)
    
    print(explanation)
    print("="*70)
    print(f"✓ All explanations saved to '{save_dir}/'")
    print("="*70)
    
    return explanation


if __name__ == '__main__':
    """
    Test interpretability module
    """
    print("Interpretability Module Test")
    print("This module requires a trained model.")
    print("\nTo use:")
    print("  1. Train model with train.py")
    print("  2. Run: python interpretability.py")
    print("  3. Explanations will be generated for sample cases")
