# Modified Code Cells for Situation → Action Training
# Copy and paste these cells into your notebook

# ============================================================================
# CELL 2: DATA LOADING (MODIFIED)
# ============================================================================

import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    get_scheduler
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import evaluate
import time
import matplotlib.pyplot as plt
import random

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# LOAD YOUR LOCAL DATASET
# ============================================================================
print("="*80)
print("LOADING LOCAL DATASET")
print("="*80)

# OPTION 1: Load from CSV
df = pd.read_csv('your_dataset.csv')

# OPTION 2: Load from Excel
# df = pd.read_excel('your_dataset.xlsx')

# Assuming columns: 'category', 'situation', 'action'
# Filter out rows with missing values
df = df.dropna(subset=['situation', 'action']).reset_index(drop=True)

# Optional: Filter by specific category
# df = df[df['category'] == 'your_category']

print(f"Total samples: {len(df)}")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nSample data:")
print(df.head(3))

# Check for data distribution by category (if you have categories)
if 'category' in df.columns:
    print(f"\nData distribution by category:")
    print(df['category'].value_counts())


# ============================================================================
# CELL 5: DATA PREPARATION (MODIFIED)
# ============================================================================

print("\n" + "="*80)
print("PREPARING SITUATION → ACTION DATA")
print("="*80)

max_input_length = 512
max_target_length = 256

def prepare_situation_action_data(df, include_category=True, use_prompt=True):
    """
    Prepare data: situation -> action
    
    Args:
        df: DataFrame with 'situation', 'action', and optionally 'category'
        include_category: Whether to include category in input
        use_prompt: Whether to use a structured prompt format
    """
    data = []
    
    for _, row in df.iterrows():
        # Option 1: Simple situation as input
        if not include_category and not use_prompt:
            input_text = row['situation']
        
        # Option 2: Include category
        elif include_category and not use_prompt:
            if 'category' in df.columns:
                input_text = f"Category: {row['category']}\nSituation: {row['situation']}"
            else:
                input_text = row['situation']
        
        # Option 3: Structured prompt (RECOMMENDED)
        elif use_prompt and not include_category:
            input_text = f"Given the situation: {row['situation']}\nWhat action should be taken?"
        
        # Option 4: Structured prompt with category (MOST CONTEXT)
        else:  # use_prompt and include_category
            if 'category' in df.columns:
                input_text = f"Category: {row['category']}\nSituation: {row['situation']}\nRecommended action:"
            else:
                input_text = f"Situation: {row['situation']}\nRecommended action:"
        
        data.append({
            'input_text': input_text,
            'target_text': row['action']
        })
    
    return data

# Choose your preferred formatting style:
# 1. Simple: include_category=False, use_prompt=False
# 2. With category: include_category=True, use_prompt=False
# 3. With prompt: include_category=False, use_prompt=True
# 4. Full context: include_category=True, use_prompt=True (RECOMMENDED)

train_data = prepare_situation_action_data(train_df, include_category=True, use_prompt=True)
val_data = prepare_situation_action_data(val_df, include_category=True, use_prompt=True)
test_data = prepare_situation_action_data(test_df, include_category=True, use_prompt=True)

print("Example formatted input:")
print(train_data[0]['input_text'])
print("\nExpected output:")
print(train_data[0]['target_text'])

print(f"\n{len(train_data)} training examples prepared")


# ============================================================================
# VERIFICATION CELL (ADD THIS AFTER CELL 5 TO VERIFY YOUR DATA)
# ============================================================================

print("\n" + "="*80)
print("DATA VERIFICATION")
print("="*80)

# Show first 3 examples
print("\nFirst 3 training examples:")
for i in range(min(3, len(train_data))):
    print(f"\n{'='*60}")
    print(f"EXAMPLE {i+1}")
    print(f"{'='*60}")
    print(f"INPUT:\n{train_data[i]['input_text']}")
    print(f"\nTARGET:\n{train_data[i]['target_text']}")

# Check input/output lengths
input_lengths = [len(tokenizer(ex['input_text'])['input_ids']) for ex in train_data[:100]]
output_lengths = [len(tokenizer(ex['target_text'])['input_ids']) for ex in train_data[:100]]

print(f"\n{'='*60}")
print("LENGTH STATISTICS (first 100 samples)")
print(f"{'='*60}")
print(f"Input tokens  - Mean: {np.mean(input_lengths):.1f}, Max: {max(input_lengths)}, Min: {min(input_lengths)}")
print(f"Output tokens - Mean: {np.mean(output_lengths):.1f}, Max: {max(output_lengths)}, Min: {min(output_lengths)}")

if max(input_lengths) > max_input_length:
    print(f"\n⚠️  WARNING: Some inputs exceed max_input_length ({max_input_length})")
    print(f"   Consider increasing max_input_length to {max(input_lengths)}")

if max(output_lengths) > max_target_length:
    print(f"\n⚠️  WARNING: Some outputs exceed max_target_length ({max_target_length})")
    print(f"   Consider increasing max_target_length to {max(output_lengths)}")


# ============================================================================
# OPTIONAL: CATEGORY-BASED ANALYSIS (IF YOU HAVE CATEGORIES)
# ============================================================================

if 'category' in df.columns:
    print("\n" + "="*80)
    print("CATEGORY-BASED ANALYSIS")
    print("="*80)
    
    train_categories = train_df['category'].value_counts()
    val_categories = val_df['category'].value_counts()
    test_categories = test_df['category'].value_counts()
    
    print("\nTrain set distribution:")
    print(train_categories)
    
    print("\nValidation set distribution:")
    print(val_categories)
    
    print("\nTest set distribution:")
    print(test_categories)
    
    # Visualize if you want
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    train_categories.plot(kind='bar', ax=axes[0], title='Training Set')
    val_categories.plot(kind='bar', ax=axes[1], title='Validation Set')
    test_categories.plot(kind='bar', ax=axes[2], title='Test Set')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# INFERENCE EXAMPLE (ADD AT THE END OF YOUR NOTEBOOK)
# ============================================================================

def predict_action(situation, category=None, model=model, tokenizer=tokenizer):
    """
    Predict action given a situation
    
    Args:
        situation: The situation description
        category: Optional category
        model: Trained model
        tokenizer: Tokenizer
    
    Returns:
        Predicted action
    """
    # Format input the same way as training data
    if category:
        input_text = f"Category: {category}\nSituation: {situation}\nRecommended action:"
    else:
        input_text = f"Situation: {situation}\nRecommended action:"
    
    # Tokenize
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=max_input_length,
        truncation=True
    ).to(device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_target_length,
            num_beams=4,
            early_stopping=True
        )
    
    # Decode
    predicted_action = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return predicted_action


# Test the model
print("\n" + "="*80)
print("TESTING MODEL PREDICTIONS")
print("="*80)

# Example 1: From test set
test_example = test_df.iloc[0]
predicted = predict_action(
    test_example['situation'],
    test_example.get('category')
)

print(f"\nTest Example 1:")
print(f"Situation: {test_example['situation']}")
if 'category' in test_df.columns:
    print(f"Category: {test_example['category']}")
print(f"\nActual Action: {test_example['action']}")
print(f"Predicted Action: {predicted}")

# Example 2: Custom input
custom_situation = "The server has been running slow for the past hour"
custom_category = "IT Support"  # Optional

predicted = predict_action(custom_situation, custom_category)

print(f"\n\nCustom Example:")
print(f"Situation: {custom_situation}")
print(f"Category: {custom_category}")
print(f"Predicted Action: {predicted}")
