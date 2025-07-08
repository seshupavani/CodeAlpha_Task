# 🌸 Iris Flower Classification

A simple machine learning project that classifies Iris flower species (Setosa, Versicolor, Virginica) using flower measurements — Sepal Length, Sepal Width, Petal Length, and Petal Width.

This project uses Python, pandas, scikit-learn, seaborn, and matplotlib to:
- 📂 Load the Iris dataset from a CSV file (from [Kaggle - Iris CSV](https://www.kaggle.com/datasets/saurabh00007/iriscsv/data))
- 🔄 Preprocess and encode species labels
- 📊 Visualize the dataset (pairplots)
- 🤖 Train and evaluate Logistic Regression and K-Nearest Neighbors (KNN) classifiers
- ✅ Print accuracy, classification reports, and show confusion matrices
- 📈 Make predictions on example data
It uses:
- Python 🐍
- Flask 🌐 (for the web server)
- Scikit-Learn 🤖 (for the ML model)
- Bootstrap 💅 (for simple styling)
---

## 📁 Dataset

- Download `Iris.csv` from [Kaggle - Iris CSV](https://www.kaggle.com/datasets/saurabh00007/iriscsv/data)
- Place `Iris.csv` in your project folder (same folder as your Python script).

---

## ⚙️ Installation

1. Clone this repository:
   ```bash
   (https://github.com/seshupavani/CodeAlpha_Task)
   cd iris-flower-classification
   
2. python -m venv .venv
# Activate for Windows:
.venv\Scripts\activate

3.Install dependencies:
pip install -r requirements.txt
pip install flask scikit-learn pandas

4.Run the app:
python app.py

📝 Example Input
Field	Example
Sepal Length (cm)	5.1
Sepal Width (cm)	3.5
Petal Length (cm)	1.4
Petal Width (cm)	0.2

Click Predict → It will show:

Prediction:
setosa

Deploy
For production, use a WSGI server like gunicorn and deploy to platforms like:

Render

Railway

Heroku

Example Procfile:
web: gunicorn app:app

📚 License
Free to use for learning and practice!

👤 Author
Kotha Seshupavani
https://github.com/seshupavani

