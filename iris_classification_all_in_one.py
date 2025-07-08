# iris_model.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# Load data
df = pd.read_csv("Iris.csv")

# Drop Id if present
if 'Id' in df.columns:
    df.drop('Id', axis=1, inplace=True)

# Encode target
le = LabelEncoder()
df['Species_encoded'] = le.fit_transform(df['Species'])

X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species_encoded']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Prediction function
def predict_species(features):
    lr_pred = log_reg.predict([features])[0]
    knn_pred = knn.predict([features])[0]
    return {
        "Logistic Regression": le.classes_[lr_pred],
        "KNN": le.classes_[knn_pred]
    }
