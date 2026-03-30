import pandas as pd
import os

base_path = os.path.dirname(os.path.abspath(__file__))
input_file = os.path.abspath(os.path.join(base_path, "..", "dataset", "GSE65391_series_matrix.txt"))
output_file = os.path.abspath(os.path.join(base_path, "..", "dataset", "GSE65391_medium.txt"))

required_probes = [
    "ILMN_1343291",
    "ILMN_1343295",
    "ILMN_1651210",
    "ILMN_1707010",
    "ILMN_1721245",
    "ILMN_1804139",
    "ILMN_1651209",
    "ILMN_1651199"
]

# Number of extra probes to add based on variance
extra_probe_count = 40

# Load full genomic file
df = pd.read_csv(input_file, sep="\t", comment="!")

# Keep only numeric sample columns for variance calculation
meta_cols = ["ID_REF"]
sample_cols = [col for col in df.columns if col not in meta_cols]

# Convert sample values to numeric
df_numeric = df.copy()
for col in sample_cols:
    df_numeric[col] = pd.to_numeric(df_numeric[col], errors="coerce")

# Compute variance across samples
df_numeric["variance"] = df_numeric[sample_cols].var(axis=1, skipna=True)

# Remove required probes before selecting top variable extras
extra_df = df_numeric[~df_numeric["ID_REF"].isin(required_probes)].copy()

# Get top variable probes
top_extra = extra_df.nlargest(extra_probe_count, "variance")

# Combine required + extra
final_df = pd.concat([
    df[df["ID_REF"].isin(required_probes)],
    df[df["ID_REF"].isin(top_extra["ID_REF"])]
]).drop_duplicates(subset=["ID_REF"])

# Save reduced file
final_df.to_csv(output_file, sep="\t", index=False)

print(f"Saved medium genomic file to: {output_file}")
print(f"Number of probes: {final_df.shape[0]}")
print(f"Columns: {final_df.shape[1]}")