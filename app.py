from flask import Flask, render_template, request
import pandas as pd
from ast import literal_eval
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# --- PREPARE THE DATA ONCE (So the website is fast) ---
file_path = r"C:\Users\pbrah\Downloads\RAW_recipes.csv"
df = pd.read_csv(file_path, nrows=1000)
df['nutrition'] = df['nutrition'].apply(literal_eval)
nutri_names = ['calories', 'total_fat', 'sugar', 'sodium', 'protein', 'sat_fat', 'carbs']
df[nutri_names] = pd.DataFrame(df['nutrition'].tolist(), index=df.index)

model = NearestNeighbors(n_neighbors=5).fit(df[nutri_names])

# --- THE WEB ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get numbers from the web form
    cals = float(request.form.get('calories'))
    prot = float(request.form.get('protein'))
    
    # Create a search query (filling other values with 0 for simplicity)
    query = [cals, 0, 0, 0, prot, 0, 0]
    
    distances, indices = model.kneighbors([query])
    results = df.iloc[indices[0]].to_dict(orient='records')
    
    return render_template('index.html', recipes=results)

if __name__ == '__main__':
    app.run(debug=True)