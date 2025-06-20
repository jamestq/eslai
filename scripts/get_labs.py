import pandas as pd

accent_files = [
    "chinese_accents.tsv",
    "hk_accents.tsv",
    "indian_accents.tsv",
    "indo_accents.tsv"
]

for accent_file in accent_files:
    accent= pd.read_csv(f"data/data/{accent_file}", sep="\t")
    accent_labs = accent[["path", "sentence"]].copy()
    accent_labs["path"] = accent_labs["path"].str.replace(".mp3", "")
    for _, row in accent_labs.iterrows():
        filename = f"data/{row["path"]}.lab"
        with open(filename, "w") as f:
            f.write(str(row['sentence']))