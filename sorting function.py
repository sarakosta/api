import pandas as pd

# Read the CSV file
df = pd.read_csv('restored plants.csv', encoding='ISO-8859-1')  # Adjust the file path and encoding if needed

# Reorder the rows based on the third column (index 2)
df_sorted = df.sort_values(by=df.columns[2], ascending=True)

# Save the reordered DataFrame to a new CSV file
df_sorted.to_csv('restored_sorted.csv', index=False)

# Read the CSV file
df = pd.read_csv('restored_animal.csv', encoding='ISO-8859-1')  # Adjust the file path and encoding if needed

# Reorder the rows based on the third column (index 2)
df_sorted = df.sort_values(by=df.columns[3], ascending=True)

# Save the reordered DataFrame to a new CSV file
df_sorted.to_csv('restored_animal_sorted.csv', index=False)