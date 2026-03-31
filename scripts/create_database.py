import pandas as pd
import sqlite3

df = pd.read_csv("data/processed/cleaned_dataset.csv")

conn = sqlite3.connect("data/processed/forum_data.db")
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS posts")

cursor.execute("""
CREATE TABLE posts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    original_text TEXT NOT NULL,
    clean_text TEXT NOT NULL,
    original_class INTEGER NOT NULL,
    binary_label INTEGER NOT NULL
)
""")

for _, row in df.iterrows():
    cursor.execute("""
        INSERT INTO posts (original_text, clean_text, original_class, binary_label)
        VALUES (?, ?, ?, ?)
    """, (
        row["tweet"],
        row["clean_text"],
        int(row["class"]),
        int(row["label"])
    ))

conn.commit()

cursor.execute("SELECT COUNT(*) FROM posts")
count = cursor.fetchone()[0]
print("Inserted rows into database:", count)

conn.close()