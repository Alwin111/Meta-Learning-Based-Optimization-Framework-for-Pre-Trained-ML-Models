# Dataset Description

## Dataset Name
CIC-IoT 2023

## Dataset Type
Large-scale tabular CSV dataset

## Task
Supervised multi-class classification

## Label Column
label

## Raw Dataset Size
~6.1 GB (CSV)

## Preprocessing Steps
- Chunk-based loading (100,000 rows per chunk)
- Sampled first 1,000,000 rows
- Removed missing values
- Feature scaling using StandardScaler
- Stratified train-test split (80/20)

## Directory Structure
- Raw data: data/raw/
- Processed data: data/processed/

## Notes
Raw and processed datasets are excluded from version control due to size.
