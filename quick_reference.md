# Quick Reference: Changes for Situation → Action Training

## 🎯 What You Need to Do

### Step 1: Prepare Your Dataset
Your CSV/Excel file should have these columns:
- `situation` (required) - the input situation
- `action` (required) - the desired action/output
- `category` (optional) - category for organization

Example:
```
category,situation,action
Safety,Fire detected in building,Evacuate all personnel and call emergency services
IT,Server down,Restart server and check logs for errors
```

### Step 2: Modify Cell 2 (Data Loading)

**FIND THIS:**
```python
dataset = load_dataset("tatsu-lab/alpaca")
df = pd.DataFrame(dataset['train'])
df = df.dropna(subset=['instruction', 'output']).reset_index(drop=True)
df['input'] = df['input'].fillna('')
```

**REPLACE WITH:**
```python
# Load your dataset
df = pd.read_csv('your_dataset.csv')  # or .read_excel()
df = df.dropna(subset=['situation', 'action']).reset_index(drop=True)
```

### Step 3: Modify Cell 5 (Data Preparation)

**FIND THIS:**
```python
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

train_data = prepare_alpaca_data(train_df)
val_data = prepare_alpaca_data(val_df)
test_data = prepare_alpaca_data(test_df)
```

**REPLACE WITH:**
```python
def prepare_situation_action_data(df):
    data = []
    for _, row in df.iterrows():
        # Choose one format:
        
        # Option A: Simple (just situation)
        # input_text = row['situation']
        
        # Option B: With category (RECOMMENDED if you have categories)
        if 'category' in df.columns:
            input_text = f"Category: {row['category']}\nSituation: {row['situation']}\nRecommended action:"
        else:
            input_text = f"Situation: {row['situation']}\nRecommended action:"
        
        data.append({
            'input_text': input_text,
            'target_text': row['action']
        })
    return data

train_data = prepare_situation_action_data(train_df)
val_data = prepare_situation_action_data(val_df)
test_data = prepare_situation_action_data(test_df)
```

### Step 4: Everything Else Stays the Same! ✅
- Cell 3 (Train/Val/Test split) - No changes
- Cell 4 (Model loading) - No changes
- Cell 6 (Tokenization) - No changes
- Cell 7+ (Training & Evaluation) - No changes

---

## 📝 Summary of Changes

| What Changed | From | To |
|--------------|------|-----|
| **Data Source** | HuggingFace Alpaca dataset | Your local CSV/Excel file |
| **Input Column** | `instruction` + `input` | `situation` (+ optional `category`) |
| **Output Column** | `output` | `action` |
| **Data Loading** | `load_dataset()` | `pd.read_csv()` or `pd.read_excel()` |
| **Preparation Function** | `prepare_alpaca_data()` | `prepare_situation_action_data()` |

---

## ✅ Checklist Before Training

- [ ] Dataset file is in correct format (situation, action columns)
- [ ] Dataset file path is correct in Cell 2
- [ ] Cell 2 modified to load your local file
- [ ] Cell 5 modified to use new preparation function
- [ ] Run cells 1-3 to verify data loads correctly
- [ ] Check sample outputs look correct
- [ ] Start training!

---

## 🔍 Verification

After Cell 5, add this to check your data:
```python
print("\nVerifying first 3 examples:")
for i in range(3):
    print(f"\n=== Example {i+1} ===")
    print(f"INPUT:  {train_data[i]['input_text'][:200]}")
    print(f"OUTPUT: {train_data[i]['target_text'][:100]}")
```

Expected output should show your situations as input and actions as output.

---

## 💡 Tips

1. **Start Small**: Test with a small dataset (100-500 samples) first
2. **Check Lengths**: Make sure situations fit in 512 tokens, actions in 256 tokens
3. **Quality Over Quantity**: Clean data is more important than lots of data
4. **Save Checkpoints**: The model auto-saves, but keep your original dataset safe!

---

## 🚨 Common Issues

**Issue**: "KeyError: 'instruction'"
**Fix**: You forgot to change Cell 5 - it's still looking for 'instruction' column

**Issue**: "FileNotFoundError"
**Fix**: Check your file path in Cell 2 - use absolute path if needed

**Issue**: Model predictions are random/bad
**Fix**: 
- Check you have enough data (1000+ samples recommended)
- Verify your data quality
- May need to train longer or adjust hyperparameters

---

That's it! Just 2 cells to modify 🎉
