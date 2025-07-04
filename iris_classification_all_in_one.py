# ------------------------------------------------------------
# iris_classification_all_in_one.py
# ------------------------------------------------------------

# 📌 Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)

# ------------------------------------------------------------
# 📌 Load the Kaggle CSV
print("\n📥 Loading Iris CSV dataset...")
df = pd.read_csv("Iris.csv")

print("\nFirst 5 rows:")
print(df.head())

# Drop Id if it exists
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

# ------------------------------------------------------------
# 📌 Encode Species labels
le = LabelEncoder()
df['Species_encoded'] = le.fit_transform(df['Species'])

print(f"\nClasses found: {le.classes_}")

# ------------------------------------------------------------
# 📌 Features & target
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species_encoded']

# ------------------------------------------------------------
# 📌 Visualize (optional)
print("\n🔍 Generating pairplot...")
try:
    sns.pairplot(df, hue='Species')
    plt.show()
except Exception as e:
    print(f"Could not generate pairplot: {e}")

# ------------------------------------------------------------
# 📌 Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {X_train.shape[0]}")
print(f"Testing samples: {X_test.shape[0]}")

# ------------------------------------------------------------
# 📌 Logistic Regression
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

print("\n✅ Logistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr, target_names=le.classes_))

ConfusionMatrixDisplay.from_estimator(
    log_reg, X_test, y_test,
    display_labels=le.classes_,
    cmap='Blues'
)
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# ------------------------------------------------------------
# 📌 KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

print("\n✅ KNN Results")
print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn, target_names=le.classes_))

ConfusionMatrixDisplay.from_estimator(
    knn, X_test, y_test,
    display_labels=le.classes_,
    cmap='Greens'
)
plt.title("KNN Confusion Matrix")
plt.show()

# ------------------------------------------------------------
# 📌 Example prediction
print("\n📌 Example prediction using one test sample:")
example = X_test.iloc[[0]]
true_label = y_test.iloc[0]
print(f"Input: {example.values}")
print(f"True: {true_label} ({le.classes_[true_label]})")
print(f"Logistic Regression predicts: {le.classes_[log_reg.predict(example)[0]]}")
print(f"KNN predicts: {le.classes_[knn.predict(example)[0]]}")

print("\n🎉 ✅ Done! Iris CSV classification complete.")
