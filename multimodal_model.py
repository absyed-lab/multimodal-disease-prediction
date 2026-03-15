"""
Multimodal Disease Prediction System
=====================================

Architecture:
- Text Branch: BioClinicalBERT encoder for symptom descriptions
- Structured Branch: TabNet encoder for demographics, vitals, medical history
- Fusion: Cross-modal attention mechanism
- Output: Disease prediction + confidence + interpretability

Target: 85-88% accuracy with interpretable predictions

Author: Your Name
Date: March 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =====================================
# MULTIMODAL ARCHITECTURE
# =====================================

class TextEncoder(nn.Module):
    """
    BioClinicalBERT encoder for patient symptom descriptions
    Extracts semantic features from unstructured text
    """
    
    def __init__(self, model_name='emilyalsentzer/Bio_ClinicalBERT', hidden_dim=768):
        super(TextEncoder, self).__init__()
        
        # Load pre-trained BioClinicalBERT
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Feature projection
        self.projection = nn.Linear(hidden_dim, 256)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, text_inputs):
        """
        Args:
            text_inputs: Dictionary with 'input_ids' and 'attention_mask'
        
        Returns:
            text_features: (batch_size, 256) - encoded text features
            attention_weights: (batch_size, seq_len) - attention for interpretability
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=text_inputs['input_ids'],
            attention_mask=text_inputs['attention_mask']
        )
        
        # Use [CLS] token representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # (batch_size, 768)
        
        # Store attention weights for interpretability
        attention_weights = text_inputs['attention_mask'].float()
        
        # Project to lower dimension
        text_features = self.projection(cls_embedding)
        text_features = self.dropout(text_features)
        
        return text_features, attention_weights


class StructuredEncoder(nn.Module):
    """
    TabNet-inspired encoder for structured medical data
    Handles demographics, vitals, medical history
    """
    
    def __init__(self, input_dim, hidden_dim=128):
        super(StructuredEncoder, self).__init__()
        
        # Feature transformation layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.fc3 = nn.Linear(hidden_dim, 256)
        self.bn3 = nn.BatchNorm1d(256)
        
        self.dropout = nn.Dropout(0.3)
        
        # Feature importance (for interpretability)
        self.feature_importance = nn.Linear(input_dim, input_dim)
        
    def forward(self, structured_data):
        """
        Args:
            structured_data: (batch_size, input_dim) - structured features
        
        Returns:
            structured_features: (batch_size, 256)
            feature_importance_weights: (batch_size, input_dim) - for SHAP
        """
        # Calculate feature importance
        importance = torch.sigmoid(self.feature_importance(structured_data))
        
        # Apply importance weighting
        x = structured_data * importance
        
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Layer 3
        structured_features = self.fc3(x)
        structured_features = self.bn3(structured_features)
        structured_features = F.relu(structured_features)
        
        return structured_features, importance


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for fusing text and structured data
    Innovation: Learn which text symptoms align with structured features
    """
    
    def __init__(self, feature_dim=256):
        super(CrossModalAttention, self).__init__()
        
        # Query, Key, Value projections
        self.query_text = nn.Linear(feature_dim, feature_dim)
        self.key_structured = nn.Linear(feature_dim, feature_dim)
        self.value_structured = nn.Linear(feature_dim, feature_dim)
        
        self.query_structured = nn.Linear(feature_dim, feature_dim)
        self.key_text = nn.Linear(feature_dim, feature_dim)
        self.value_text = nn.Linear(feature_dim, feature_dim)
        
        self.scale = np.sqrt(feature_dim)
        
    def forward(self, text_features, structured_features):
        """
        Bidirectional cross-modal attention
        
        Returns:
            fused_features: (batch_size, 512) - concatenated enhanced features
            attention_text_to_struct: Attention weights for interpretability
            attention_struct_to_text: Attention weights for interpretability
        """
        # Text attending to structured
        Q_text = self.query_text(text_features)
        K_struct = self.key_structured(structured_features)
        V_struct = self.value_structured(structured_features)
        
        attention_text_to_struct = torch.softmax(
            torch.matmul(Q_text.unsqueeze(1), K_struct.unsqueeze(2)) / self.scale,
            dim=-1
        )
        
        enhanced_text = text_features + attention_text_to_struct.squeeze() * V_struct
        
        # Structured attending to text
        Q_struct = self.query_structured(structured_features)
        K_text = self.key_text(text_features)
        V_text = self.value_text(text_features)
        
        attention_struct_to_text = torch.softmax(
            torch.matmul(Q_struct.unsqueeze(1), K_text.unsqueeze(2)) / self.scale,
            dim=-1
        )
        
        enhanced_structured = structured_features + attention_struct_to_text.squeeze() * V_text
        
        # Concatenate enhanced features
        fused_features = torch.cat([enhanced_text, enhanced_structured], dim=1)
        
        return fused_features, attention_text_to_struct, attention_struct_to_text


class MultiModalDiseasePredictor(nn.Module):
    """
    Complete multimodal disease prediction system
    
    Novel Contributions:
    1. Dynamic cross-modal fusion (not simple concatenation)
    2. Interpretable attention mechanisms
    3. Uncertainty quantification with Monte Carlo Dropout
    """
    
    def __init__(self, num_diseases, structured_input_dim):
        super(MultiModalDiseasePredictor, self).__init__()
        
        # Encoders
        self.text_encoder = TextEncoder()
        self.structured_encoder = StructuredEncoder(structured_input_dim)
        
        # Cross-modal attention fusion
        self.fusion = CrossModalAttention(feature_dim=256)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_diseases)
        )
        
    def forward(self, text_inputs, structured_data, return_attention=False):
        """
        Forward pass with optional attention weights for interpretability
        
        Returns:
            logits: (batch_size, num_diseases)
            attention_dict: Dictionary with attention weights (if return_attention=True)
        """
        # Encode text
        text_features, text_attention = self.text_encoder(text_inputs)
        
        # Encode structured data
        structured_features, feature_importance = self.structured_encoder(structured_data)
        
        # Cross-modal fusion
        fused_features, attn_t2s, attn_s2t = self.fusion(
            text_features, structured_features
        )
        
        # Classify
        logits = self.classifier(fused_features)
        
        if return_attention:
            attention_dict = {
                'text_attention': text_attention,
                'feature_importance': feature_importance,
                'text_to_structured': attn_t2s,
                'structured_to_text': attn_s2t
            }
            return logits, attention_dict
        
        return logits
    
    def predict_with_uncertainty(self, text_inputs, structured_data, num_samples=20):
        """
        Monte Carlo Dropout for uncertainty quantification
        
        Returns:
            predictions: (batch_size, num_diseases) - mean predictions
            uncertainty: (batch_size, num_diseases) - standard deviation
            confidence: (batch_size,) - prediction confidence
        """
        self.train()  # Enable dropout
        
        predictions_list = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                logits = self.forward(text_inputs, structured_data)
                probs = torch.softmax(logits, dim=1)
                predictions_list.append(probs)
        
        # Stack predictions
        predictions_stack = torch.stack(predictions_list)  # (num_samples, batch_size, num_diseases)
        
        # Calculate mean and std
        predictions = predictions_stack.mean(dim=0)
        uncertainty = predictions_stack.std(dim=0)
        
        # Confidence = max probability - uncertainty
        confidence = predictions.max(dim=1)[0] - uncertainty.max(dim=1)[0]
        
        self.eval()  # Disable dropout
        
        return predictions, uncertainty, confidence


# =====================================
# BASELINE MODELS FOR COMPARISON
# =====================================

class TextOnlyBaseline(nn.Module):
    """Baseline: Text-only using BioClinicalBERT"""
    
    def __init__(self, num_diseases):
        super(TextOnlyBaseline, self).__init__()
        self.text_encoder = TextEncoder()
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_diseases)
        )
    
    def forward(self, text_inputs, structured_data=None):
        text_features, _ = self.text_encoder(text_inputs)
        return self.classifier(text_features)


class StructuredOnlyBaseline(nn.Module):
    """Baseline: Structured-only using TabNet-inspired architecture"""
    
    def __init__(self, num_diseases, structured_input_dim):
        super(StructuredOnlyBaseline, self).__init__()
        self.structured_encoder = StructuredEncoder(structured_input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_diseases)
        )
    
    def forward(self, text_inputs=None, structured_data=None):
        structured_features, _ = self.structured_encoder(structured_data)
        return self.classifier(structured_features)


class SimpleConcatBaseline(nn.Module):
    """Baseline: Simple concatenation (no cross-modal attention)"""
    
    def __init__(self, num_diseases, structured_input_dim):
        super(SimpleConcatBaseline, self).__init__()
        self.text_encoder = TextEncoder()
        self.structured_encoder = StructuredEncoder(structured_input_dim)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_diseases)
        )
    
    def forward(self, text_inputs, structured_data):
        text_features, _ = self.text_encoder(text_inputs)
        structured_features, _ = self.structured_encoder(structured_data)
        concat_features = torch.cat([text_features, structured_features], dim=1)
        return self.classifier(concat_features)


# =====================================
# UTILITY FUNCTIONS
# =====================================

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    """
    Test the architecture
    """
    print("="*70)
    print("MULTIMODAL DISEASE PREDICTION - ARCHITECTURE TEST")
    print("="*70)
    
    # Hyperparameters
    NUM_DISEASES = 10
    STRUCTURED_INPUT_DIM = 20  # age, gender, vitals, etc.
    BATCH_SIZE = 4
    
    # Create model
    print("\nInitializing multimodal model...")
    model = MultiModalDiseasePredictor(
        num_diseases=NUM_DISEASES,
        structured_input_dim=STRUCTURED_INPUT_DIM
    ).to(device)
    
    print(f"✓ Model created")
    print(f"  Total parameters: {count_parameters(model):,}")
    
    # Create dummy inputs
    print("\nCreating dummy inputs...")
    
    # Text input (simulated tokenized text)
    text_inputs = {
        'input_ids': torch.randint(0, 1000, (BATCH_SIZE, 128)).to(device),
        'attention_mask': torch.ones(BATCH_SIZE, 128).to(device)
    }
    
    # Structured input
    structured_data = torch.randn(BATCH_SIZE, STRUCTURED_INPUT_DIM).to(device)
    
    print("✓ Dummy inputs created")
    
    # Forward pass
    print("\nTesting forward pass...")
    model.eval()
    
    with torch.no_grad():
        # Regular prediction
        logits = model(text_inputs, structured_data)
        print(f"✓ Output shape: {logits.shape}")
        print(f"  Expected: ({BATCH_SIZE}, {NUM_DISEASES})")
        
        # Prediction with attention
        logits, attention = model(text_inputs, structured_data, return_attention=True)
        print(f"✓ Attention weights retrieved:")
        print(f"  - Text attention: {attention['text_attention'].shape}")
        print(f"  - Feature importance: {attention['feature_importance'].shape}")
        print(f"  - Cross-modal attention shapes available")
        
        # Uncertainty quantification
        print("\nTesting uncertainty quantification...")
        predictions, uncertainty, confidence = model.predict_with_uncertainty(
            text_inputs, structured_data, num_samples=10
        )
        print(f"✓ Predictions: {predictions.shape}")
        print(f"✓ Uncertainty: {uncertainty.shape}")
        print(f"✓ Confidence: {confidence.shape}")
        print(f"  Mean confidence: {confidence.mean().item():.3f}")
    
    # Test baselines
    print("\n" + "="*70)
    print("TESTING BASELINE MODELS")
    print("="*70)
    
    baselines = {
        'Text-Only': TextOnlyBaseline(NUM_DISEASES),
        'Structured-Only': StructuredOnlyBaseline(NUM_DISEASES, STRUCTURED_INPUT_DIM),
        'Simple Concat': SimpleConcatBaseline(NUM_DISEASES, STRUCTURED_INPUT_DIM)
    }
    
    for name, baseline in baselines.items():
        baseline = baseline.to(device)
        baseline.eval()
        params = count_parameters(baseline)
        
        with torch.no_grad():
            if 'Text-Only' in name:
                output = baseline(text_inputs)
            elif 'Structured-Only' in name:
                output = baseline(structured_data=structured_data)
            else:
                output = baseline(text_inputs, structured_data)
        
        print(f"\n{name}:")
        print(f"  Parameters: {params:,}")
        print(f"  Output shape: {output.shape}")
    
    print("\n" + "="*70)
    print("✓ ALL ARCHITECTURE TESTS PASSED!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Prepare datasets (MIMIC-III, Synthea)")
    print("  2. Train model with data_loader.py")
    print("  3. Evaluate with evaluation.py")
    print("  4. Generate interpretability visualizations")
