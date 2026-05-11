This branch contains only the data processing part of the project.

How to run:

# 1) Inspect raw dataset (shape, columns, missing values, class distribution)
python scripts/inspect_data.py

# 2) Clean text, map labels, balance classes, and save processed dataset
python scripts/clean_data.py

# 3) Create and populate SQLite database from processed dataset
python scripts/create_database.py

# 4) Split processed dataset into train/validation/test files
python scripts/split_data.py