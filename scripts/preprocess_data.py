import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path

RAW_PATH = Path("data/raw/ciciot-2023.csv")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "label"
CHUNK_SIZE = 100_000
MAX_ROWS = 1_000_000   # 1 million rows (safe & sufficient)

X_chunks = []
y_chunks = []
rows_read = 0

print("Starting chunk-based preprocessing...")

for chunk in pd.read_csv(RAW_PATH, chunksize=CHUNK_SIZE):
    chunk = chunk.dropna()

    X = chunk.drop(columns=[TARGET_COLUMN])
    y = chunk[TARGET_COLUMN]

    X_chunks.append(X)
    y_chunks.append(y)

    rows_read += len(chunk)
    print(f"Processed {rows_read} rows")

    if rows_read >= MAX_ROWS:
        break

print("Concatenating chunks...")
X_full = pd.concat(X_chunks)
y_full = pd.concat(y_chunks)

print("Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)

print("Splitting train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_full,
    test_size=0.2,
    random_state=42,
    stratify=y_full
)

print("Saving processed files...")
pd.DataFrame(X_train).to_csv(PROCESSED_DIR / "X_train.csv", index=False)
pd.DataFrame(X_test).to_csv(PROCESSED_DIR / "X_test.csv", index=False)
pd.DataFrame(y_train).to_csv(PROCESSED_DIR / "y_train.csv", index=False)
pd.DataFrame(y_test).to_csv(PROCESSED_DIR / "y_test.csv", index=False)

print("✅ Preprocessing complete. Files saved in data/processed/")
