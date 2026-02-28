import pandas as pd
from ast import literal_eval
from sklearn.neighbors import NearestNeighbors # The ML Brain

file_path = r"C:\Users\pbrah\Downloads\RAW_recipes.csv"

# 1. Load 1000 rows this time (more options for the model)
df = pd.read_csv(file_path, nrows=1000)

# 2. Clean the data (same as before)
df['nutrition'] = df['nutrition'].apply(literal_eval)
nutri_names = ['calories', 'total_fat', 'sugar', 'sodium', 'protein', 'sat_fat', 'carbs']
df[nutri_names] = pd.DataFrame(df['nutrition'].tolist(), index=df.index)

# 3. SET UP THE MACHINE LEARNING MODEL
# We tell the model to look ONLY at the nutrition columns
features = df[nutri_names]
model = NearestNeighbors(n_neighbors=3, algorithm='ball_tree')
model.fit(features)

# 4. TEST THE RECOMMENDATION
# Imagine a user wants: 300 cals, 10 fat, 5 sugar, 5 sodium, 20 protein, 5 sat_fat, 10 carbs
user_input = [300, 10, 5, 5, 20, 5, 10]

distances, indices = model.kneighbors([user_input])

print("✨ Top 3 Recommended Recipes for you: ✨")
print(df.iloc[indices[0]][['name', 'calories', 'protein']])