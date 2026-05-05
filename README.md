Commands to run the code: 

# shows us what the raw data looks like before touching it. how many rows/columns, if anything misses and how many tweets beong # to each class (0,1,2)
python scripts/inspect_data.py 

# confirms how many rows survived cleaning, shows the balance between label 0 and label 1 and confirms the file was saved
python scripts/clean_data.py

# confirms how many rows were successfully inserted into the SQLite database
python scripts/create_database.py

# confirms the size of each split so we know the training, validation and test sets were created correctly
scripts/split_data.py