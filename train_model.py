import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import os

def train_sentiment_model():
    """
    Fine-tunes the mental-bert model on your custom annotated dataset.
    """
    print("🧠 Starting Custom Sentiment Model Training...")
    print("=" * 50)

    # --- 1. Load Your Annotated Data ---
    annotations_path = "data/annotations.csv"
    try:
        df = pd.read_csv(annotations_path)
        if 'text' not in df.columns or 'label' not in df.columns:
            print(f"❌ Error: '{annotations_path}' must have 'text' and 'label' columns.")
            return
    except FileNotFoundError:
        print(f"❌ Error: '{annotations_path}' not found. Please run 'label_data.py' first.")
        return

    labels = sorted(df['label'].unique().tolist())
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    
    df['label_id'] = df['label'].map(label2id)

    print(f"✅ Loaded {len(df)} annotated examples.")
    print(f"🏷️  Labels found: {labels}")

    # --- 2. Prepare the Dataset ---
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    train_dataset = Dataset.from_pandas(train_df)
    eval_dataset = Dataset.from_pandas(eval_df)
    
    # --- 3. Tokenize the Data ---
    model_name = "mental/mental-bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    
    # --- MODIFICATION START ---
    # Remove the old columns that are not needed for training to avoid confusion
    train_dataset = train_dataset.remove_columns(['text', 'label', '__index_level_0__'])
    eval_dataset = eval_dataset.remove_columns(['text', 'label', '__index_level_0__'])
    
    # Rename the label_id column to 'labels' which is the expected name for the Trainer
    train_dataset = train_dataset.rename_column("label_id", "labels")
    eval_dataset = eval_dataset.rename_column("label_id", "labels")
    # --- MODIFICATION END ---

    # --- 4. Load the Model ---
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id
    )

    # --- 5. Set Up the Trainer ---
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=10,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # --- 6. Train the Model ---
    print("\n🚀 Starting model fine-tuning... This will take a few minutes.")
    trainer.train()

    # --- 7. Save the Model ---
    output_model_path = "./my_custom_mental_health_model"
    trainer.save_model(output_model_path)
    tokenizer.save_pretrained(output_model_path)
    
    print("\n" + "=" * 50)
    print("🎉 Training Complete!")
    print(f"💾 Model saved to: {output_model_path}")

if __name__ == "__main__":
    train_sentiment_model()