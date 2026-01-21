# Guide: Adapting Your Model for Category | Situation | Action Dataset

## Overview
Your current notebook trains a model on the Alpaca dataset (instruction → output format).
You want to adapt it to train on your local dataset with format: **Category | Situation | Action**

The model should predict **Action** given **Situation** as input.

---

## Key Changes Required

### 1. **DATA LOADING** (Cell 2)
**Current code:**
```python
# Load Alpaca dataset from Hugging Face
dataset = load_dataset("tatsu-lab/alpaca")

# Convert to pandas for easier manipulation
df = pd.DataFrame(dataset['train'])

# Filter out rows with missing values
df = df.dropna(subset=['instruction', 'output']).reset_index(drop=True)

# Fill empty 'input' fields with empty string
df['input'] = df['input'].fillna('')
```

**Replace with:**
```python
# Load your local CSV/Excel file
import pandas as pd

# Choose one based on your file format:
df = pd.read_csv('your_dataset.csv')  # For CSV files
# OR
df = pd.read_excel('your_dataset.xlsx')  # For Excel files

# Assuming your columns are: 'category', 'situation', 'action'
# Filter out rows with missing values in the columns we need
df = df.dropna(subset=['situation', 'action']).reset_index(drop=True)

# Optional: You can also filter by specific categories if needed
# df = df[df['category'] == 'specific_category']
```

**Important:** Make sure your dataset file has these column names:
- `category` (optional, can be used for filtering)
- `situation` (the input/context)
- `action` (the target/output to predict)

---

### 2. **DATA PREPARATION** (Cell 5)
**Current code:**
```python
def prepare_alpaca_data(df):
    """
    Prepare Alpaca format data: (instruction + input) -> output
    """
    data = []
    for _, row in df.iterrows():
        # Combine instruction and input
        if row['input'].strip():
            input_text = f"{row['instruction']}\n\nInput: {row['input']}"
        else:
            input_text = row['instruction']

        data.append({
            'input_text': input_text,
            'target_text': row['output']
        })
    return data
```

**Replace with:**
```python
def prepare_situation_action_data(df):
    """
    Prepare data: situation -> action
    You can optionally include category in the input
    """
    data = []
    for _, row in df.iterrows():
        # Option 1: Use only situation as input
        input_text = row['situation']
        
        # Option 2: Include category in the input for better context
        # input_text = f"Category: {row['category']}\n\nSituation: {row['situation']}"
        
        # Option 3: More structured prompt (recommended)
        # input_text = f"Given the situation: {row['situation']}\nWhat action should be taken?"

        data.append({
            'input_text': input_text,
            'target_text': row['action']
        })
    return data

# Update the function calls
train_data = prepare_situation_action_data(train_df)
val_data = prepare_situation_action_data(val_df)
test_data = prepare_situation_action_data(test_df)
```

**Choose the option that works best:**
- **Option 1**: Simplest, just feeds the situation
- **Option 2**: Includes category for additional context
- **Option 3**: Creates a more natural instruction-following format

---

## Complete Example of Changes

Here's a minimal example showing before/after for the key sections:

### BEFORE (Alpaca format):
```python
# Cell 2: Load data
dataset = load_dataset("tatsu-lab/alpaca")
df = pd.DataFrame(dataset['train'])

# Cell 5: Prepare data
def prepare_alpaca_data(df):
    data = []
    for _, row in df.iterrows():
        if row['input'].strip():
            input_text = f"{row['instruction']}\n\nInput: {row['input']}"
        else:
            input_text = row['instruction']
        data.append({
            'input_text': input_text,
            'target_text': row['output']
        })
    return data
```

### AFTER (Your format):
```python
# Cell 2: Load data
df = pd.read_csv('/path/to/your/dataset.csv')
df = df.dropna(subset=['situation', 'action']).reset_index(drop=True)

# Cell 5: Prepare data
def prepare_situation_action_data(df):
    data = []
    for _, row in df.iterrows():
        # Recommended: structured prompt
        input_text = f"Situation: {row['situation']}\nWhat action should be taken?"
        
        data.append({
            'input_text': input_text,
            'target_text': row['action']
        })
    return data

train_data = prepare_situation_action_data(train_df)
val_data = prepare_situation_action_data(val_df)
test_data = prepare_situation_action_data(test_df)
```

---

## Summary of Cell Changes

| Cell Number | Description | Action Required |
|-------------|-------------|-----------------|
| **Cell 2** | Data Loading | Replace `load_dataset()` with `pd.read_csv()` or `pd.read_excel()` |
| **Cell 3** | Data Splitting | No changes needed ✓ |
| **Cell 4** | Model Loading | No changes needed ✓ |
| **Cell 5** | Data Preparation | Replace `prepare_alpaca_data()` with `prepare_situation_action_data()` |
| **Cell 6** | Tokenization | No changes needed ✓ |
| **Cell 7+** | Training/Evaluation | No changes needed ✓ |

---

## Example Dataset Format

Your CSV/Excel file should look like this:

```csv
category,situation,action
Safety,Fire alarm triggered in building,Evacuate all personnel immediately
Customer Service,Customer complaint about product quality,Apologize and offer replacement or refund
IT Support,Computer won't boot up,Check power connections and restart
```

---

## Testing Your Changes

After making the changes, test with a few samples:

```python
# After Cell 5, add this to verify your data
print("\nFirst 3 training examples:")
for i in range(min(3, len(train_data))):
    print(f"\n--- Example {i+1} ---")
    print(f"INPUT: {train_data[i]['input_text']}")
    print(f"TARGET: {train_data[i]['target_text']}")
```

This will help you verify that the data is formatted correctly before training.

---

## Additional Tips

1. **Dataset Size**: Make sure you have enough data (at least 1000-5000 examples for good results)

2. **Text Length**: Check that your situations and actions fit within the token limits:
   - `max_input_length = 512` (can be adjusted)
   - `max_target_length = 256` (can be adjusted)

3. **Categories**: If you have multiple categories, you might want to:
   - Train separate models per category
   - Include category in the input prompt
   - Analyze performance by category

4. **Data Quality**: Clean your data:
   - Remove duplicates
   - Fix typos and formatting
   - Ensure consistency in action descriptions

---

Good luck with your training! 🚀
