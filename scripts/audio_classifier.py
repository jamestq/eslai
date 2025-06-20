from datasets import load_dataset, Features, Value

features = Features({
    "client_id": Value("string"),
    "path": Value("string"),
    "sentence_id": Value("string"),
    "age": Value("string"),
    "gender": Value("string"),
    "accents": Value("string"),
})

train = load_dataset("csv", data_files="train.tsv", sep="\t", features=features)