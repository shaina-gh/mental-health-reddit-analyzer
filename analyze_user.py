import pandas as pd
import os

def analyze_user_in_thread():
    """
    Extracts all comments from a single user within a specific thread
    from an existing conversation CSV file.
    """
    print("👤 Single User Trajectory Analyzer")
    print("=" * 40)

    # --- 1. Select the Data File ---
    data_dir = "data"
    if not os.path.exists(data_dir) or not os.listdir(data_dir):
        print(f"❌ Error: The '{data_dir}' directory is empty or does not exist.")
        print("Please run the 'thread_conversation_extractor.py' script first to generate data.")
        return

    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"❌ Error: No CSV files found in the '{data_dir}' directory.")
        return

    print("Available data files:")
    for i, filename in enumerate(csv_files):
        print(f"  {i + 1}: {filename}")

    try:
        choice = int(input(f"Select a file to analyze (1-{len(csv_files)}): ")) - 1
        input_csv_path = os.path.join(data_dir, csv_files[choice])
    except (ValueError, IndexError):
        print("❌ Invalid selection. Exiting.")
        return

    print(f"\nLoading data from '{input_csv_path}'...")
    df = pd.read_csv(input_csv_path)

    # --- 2. Get User Input ---
    thread_id = input("Enter the thread_id to analyze: ").strip()
    username = input("Enter the username to track: ").strip()

    # --- 3. Filter the Data ---
    print(f"\n🔍 Searching for user '{username}' in thread '{thread_id}'...")
    user_df = df[(df['thread_id'] == thread_id) & (df['author'] == username)].copy()

    # --- 4. Display and Save Results ---
    if user_df.empty:
        print("\n❌ No comments found for this user in the specified thread.")
        print("Please check if the thread_id and username are correct.")
    else:
        user_df = user_df.sort_values(by='position')
        
        print(f"\n✅ Found {len(user_df)} posts from '{username}':")
        for index, row in user_df.iterrows():
            print("-" * 20)
            print(f"Position: {row['position']}")
            print(f"Timestamp: {row['timestamp']}")
            print(f"Text: {row['text'][:200]}...")

        # --- MODIFICATION START ---
        # Define the subfolder for user-specific analysis
        output_dir = os.path.join("data", "user_analysis")
        os.makedirs(output_dir, exist_ok=True) # Create the folder if it doesn't exist

        # Create the filename and join it with the new path
        output_filename = f"user_{username}_in_thread_{thread_id}.csv"
        output_path = os.path.join(output_dir, output_filename)
        # --- MODIFICATION END ---
        
        user_df.to_csv(output_path, index=False)

        print("\n" + "=" * 40)
        print("🎉 Analysis Complete!")
        print(f"💾 User-specific conversation saved to: {output_path}")

if __name__ == "__main__":
    analyze_user_in_thread()