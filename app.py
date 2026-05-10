from flask import Flask, render_template, request
import pandas as pd
from ast import literal_eval
from sklearn.neighbors import NearestNeighbors
import random

app = Flask(__name__)

# --- PREPARE THE DATA ---
file_path = "RAW_recipes.csv"
# We load the essential columns: name, nutrition, steps, and ingredients
df = pd.read_csv(file_path, nrows=1000)

# Clean nutrition
df['nutrition'] = df['nutrition'].apply(literal_eval)
nutri_names = ['calories', 'total_fat', 'sugar', 'sodium', 'protein', 'sat_fat', 'carbs']
df[nutri_names] = pd.DataFrame(df['nutrition'].tolist(), index=df.index)

# Fit the AI model
model = NearestNeighbors().fit(df[nutri_names])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    cals = float(request.form.get('calories'))
    prot = float(request.form.get('protein'))
    
    query = [cals, 0, 0, 0, prot, 0, 0]
    
    # AI finds 20 candidates
    distances, indices = model.kneighbors([query], n_neighbors=20)
    
    # Pick 5 random ones
    candidate_list = indices[0].tolist()
    random_selection = random.sample(candidate_list, 5)
    
    # --- NEW: Pull EXTRA columns (steps & ingredients) ---
    results = df.iloc[random_selection][['name', 'calories', 'protein', 'steps', 'ingredients']].to_dict(orient='records')
    
    # --- NEW: Cleanup logic so it looks like a real recipe ---
    for recipe in results:
        # Convert "['step 1', 'step 2']" into "Step 1. Step 2."
        steps_list = literal_eval(recipe['steps'])
        recipe['steps'] = ". ".join(steps_list).capitalize()
        
        # Convert "['salt', 'water']" into "salt, water"
        ingred_list = literal_eval(recipe['ingredients'])
        recipe['ingredients'] = ", ".join(ingred_list)
    
    return render_template('index.html', recipes=results)

if __name__ == '__main__':
    app.run(debug=True)