import pandas as pd
import os

def select(df, col, val):
    return df[df[col] == val]

# Helper functions for processing accents
def get_accent_data(df, accent_pattern):
    """Filter dataframe for rows containing the specified accent pattern."""
    return df[df["accents"].str.contains(accent_pattern, case=False)]

def create_balanced_dataset(accent_df, n_samples=150, random_state=42):
    """Create gender-balanced dataset from the given accent dataframe."""
    male_df = accent_df[accent_df["gender"].str.contains("male")]
    female_df = accent_df[accent_df["gender"].str.contains("female")]
    
    # Sample equal numbers from each gender
    male_samples = male_df.sample(n=n_samples, random_state=random_state)
    female_samples = female_df.sample(n=n_samples, random_state=random_state)
    
    # Combine and reset index
    return pd.concat([male_samples, female_samples]).reset_index(drop=True)

def process_accent(accents_df, accent_pattern, output_filename, balance=True, n_samples=150):
    """Process an accent, optionally creating a gender-balanced dataset, and save to file."""
    accent_df = get_accent_data(accents_df, accent_pattern)
    
    if balance:
        accent_df = create_balanced_dataset(accent_df, n_samples)
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Save to file
    output_path = os.path.join("data", output_filename)
    accent_df.to_csv(output_path, sep="\t")
    return output_path

def main():
    # Preprocess the training data
    train = pd.read_csv("train.tsv", sep="\t")
    train_reduced = train.drop(["client_id", "sentence_id"], axis=1)
    train_accents = train_reduced[train_reduced[["accents", "gender"]].notnull().all(1)].copy()
    train_accents["accents"] = train_accents["accents"].str.lower()
    train_accents.to_csv("data/train_accents.tsv", sep="\t")

    # Retrieving specific accent
    accents = pd.read_csv("data/train_accents.tsv", sep="\t")
    
    # Process each accent type
    accent_files = {
        "indian": process_accent(accents, "india and south asia", "indian_accents.tsv"),
        "chinese": process_accent(accents, "chinese", "chinese_accents.tsv", balance=False),
        "hk": process_accent(accents, "hong kong", "hk_accents.tsv"),
        "indonesian": process_accent(accents, "indonesia", "indo_accents.tsv", balance=False)
    }
    
    # Combine all accent datasets
    accent_dfs = []
    for accent_name, file_path in accent_files.items():
        df = pd.read_csv(file_path, sep="\t")
        accent_dfs.append(df)
    
    combined = pd.concat(accent_dfs).reset_index(drop=True)
    
    # Create audio paths file
    path = combined['path'].copy()
    path = "cv-corpus-21.0-2025-03-14/en/clips/" + path
    path.to_csv("audio_paths.txt", index=False, header=False)

if __name__ == "__main__":
    main()