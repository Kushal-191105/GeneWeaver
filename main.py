import pandas as pd

INPUT_FILE = "data/human_sequences.txt"

data = pd.read_csv(INPUT_FILE, sep="\t")

valid_bases = set("ATGC")
ambiguous_bases = set("N")


def classify_sequence(sequence):
    sequence = str(sequence).upper().strip()
    characters = set(sequence)

    if characters.issubset(valid_bases):
        return "valid"

    if characters.issubset(valid_bases | ambiguous_bases):
        return "ambiguous"

    return "invalid"


data["status"] = data["sequence"].apply(classify_sequence)

print("Total sequences:", len(data))
print("Valid sequences:", (data["status"] == "valid").sum())
print("Ambiguous sequences:", (data["status"] == "ambiguous").sum())
print("Invalid sequences:", (data["status"] == "invalid").sum())


# Sequence length analysis
data["length"] = data["sequence"].astype(str).str.len()

print("\nSequence Length Statistics:")
print("Minimum length:", data["length"].min())
print("Maximum length:", data["length"].max())
print("Average length:", round(data["length"].mean(), 2))
print("Total bases:", data["length"].sum())