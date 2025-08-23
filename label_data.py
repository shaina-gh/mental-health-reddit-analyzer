import pandas as pd
import os
import glob

def create_annotations_file():
    """
    An interactive command-line tool to quickly label conversation data.
    """
    # --- Configuration ---
    LABELS = [
        "Seeking-Support",
        "Offering-Support",
        "Expressing-Hope",
        "Expressing-Pain",
        "Neutral-Sharing",
    ]
    
    DATA_DIR = "data"
    ANNOTATIONS_FILE = os.path.join(DATA_DIR, "annotations.csv")
    
    print("🏷️  Interactive Data Labeling Tool")
    print("=" * 40)

    # --- 1. Find the latest raw data file ---
    list_of_files = glob.glob(os.path.join(DATA_DIR, 'mental_health_conversations_*.csv'))
    if not list_of_files:
        print(f"❌ Error: No conversation CSV files found in '{DATA_DIR}'.")
        print("Please run the 'thread_conversation_extractor.py' script first.")
        return

    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"✅ Loading data from latest file: {os.path.basename(latest_file)}\n")
    df_raw = pd.read_csv(latest_file)

    # --- 2. Load existing annotations to allow resuming ---
    if os.path.exists(ANNOTATIONS_FILE):
        df_annotated = pd.read_csv(ANNOTATIONS_FILE)
        annotated_texts = set(df_annotated['text'].tolist())
        print(f"Resuming session. Found {len(annotated_texts)} previously labeled comments.")
        new_annotations = df_annotated.to_dict('records')
    else:
        annotated_texts = set()
        new_annotations = []
        print("Starting a new labeling session.")

    # --- 3. Start the Interactive Labeling Loop ---
    texts_to_label = df_raw.dropna(subset=['text'])
    
    try:
        for index, row in texts_to_label.iterrows():
            text_to_check = row['text']
            
            if text_to_check in annotated_texts:
                continue

            os.system('cls' if os.name == 'nt' else 'clear')
            
            print(f"Progress: {len(annotated_texts)} labeled | Total in file: {len(texts_to_label)}")
            print("-" * 40)
            print("COMMENT TO LABEL:")
            print(f"'{text_to_check}'")
            print("-" * 40)
            
            print("Please choose a label:")
            for i, label in enumerate(LABELS):
                print(f"  [{i + 1}] {label}")
            print("\n  [s] Skip this comment")
            print("  [q] Quit and Save")
            
            while True:
                choice = input("Enter your choice (number, 's', or 'q'): ").strip().lower()
                if choice == 'q':
                    raise KeyboardInterrupt
                elif choice == 's':
                    break
                elif choice.isdigit() and 1 <= int(choice) <= len(LABELS):
                    selected_label = LABELS[int(choice) - 1]
                    new_annotations.append({'text': text_to_check, 'label': selected_label})
                    annotated_texts.add(text_to_check)
                    break
                else:
                    print("❌ Invalid input. Please enter a valid number, 's', or 'q'.")

    except KeyboardInterrupt:
        print("\n\n👋 Quitting and saving progress...")

    finally:
        if new_annotations:
            df_to_save = pd.DataFrame(new_annotations)
            df_to_save.to_csv(ANNOTATIONS_FILE, index=False)
            print(f"\n🎉 Saved {len(df_to_save)} labeled comments to '{ANNOTATIONS_FILE}'")
        else:
            print("\nNo new labels were added.")

if __name__ == "__main__":
    create_annotations_file()