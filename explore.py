import pandas as pd
from ast import literal_eval
from sklearn.neighbors import NearestNeighbors
import random

# 1. Load Data
file_path = "RAW_recipes.csv"
df = pd.read_csv(file_path, nrows=1000)

# 2. Clean Nutrition Data
df['nutrition'] = df['nutrition'].apply(literal_eval)
nutri_names = ['calories', 'total_fat', 'sugar', 'sodium', 'protein', 'sat_fat', 'carbs']
df[nutri_names] = pd.DataFrame(df['nutrition'].tolist(), index=df.index)

# 3. Setup Model
features = df[nutri_names]
model = NearestNeighbors(algorithm='ball_tree')
model.fit(features)

# 4. Test Input (300 Calories, 20g Protein)
user_input = [300, 10, 5, 5, 20, 5, 10]

# 5. Get 20 candidates and pick 3 random ones
distances, indices = model.kneighbors([user_input], n_neighbors=20)
random_indices = random.sample(list(indices[0]), 3)

print("\n" + "="*50)
print("✨ YOUR PERSONALIZED MEAL PLAN ✨")
print("="*50 + "\n")

# 6. Display Detailed Results
for idx in random_indices:
    row = df.iloc[idx]
    
    # Clean up the text data for the terminal
    steps = ". ".join(literal_eval(row['steps'])).capitalize()
    ingredients = ", ".join(literal_eval(row['ingredients']))
    
    print(f"🍴 RECIPE: {row['name'].upper()}")
    print(f"📊 NUTRITION: {row['calories']} Cals | {row['protein']}g Protein")
    print(f"🛒 INGREDIENTS: {ingredients}")
    print(f"👨‍🍳 STEPS: {steps}")
    print("-" * 50 + "\n")