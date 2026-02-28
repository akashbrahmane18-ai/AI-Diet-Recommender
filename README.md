# 🍲 AI Diet Recommender
An AI-powered web application that uses **Machine Learning (K-Nearest Neighbors)** to recommend recipes based on a user's calorie and protein goals.

## 🚀 Features
* **ML Engine:** Uses Scikit-Learn to find similar recipes in a 230k+ dataset.
* **Web UI:** Built with Flask and styled with Bootstrap for a modern look.
* **Data-Driven:** Analyzes nutritional content (Calories, Protein, Carbs, Fat).

## 🛠️ Tech Stack
* **Language:** Python 3.10
* **Libraries:** Pandas, Scikit-Learn, Flask
* **Frontend:** HTML5, Bootstrap 5

## 📋 How to Run
1. Download the dataset from [Kaggle (Food.com)](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions).
2. Update the `file_path` in `app.py` to point to your CSV.
3. Run `pip install flask pandas scikit-learn`.
4. Run `python app.py`.