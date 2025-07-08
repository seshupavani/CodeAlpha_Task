# app.py

from flask import Flask, render_template, request
from iris_model import predict_species

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        try:
            sl = float(request.form["sepal_length"])
            sw = float(request.form["sepal_width"])
            pl = float(request.form["petal_length"])
            pw = float(request.form["petal_width"])
            result = predict_species([sl, sw, pl, pw])
        except Exception as e:
            result = {"Logistic Regression": f"Error: {e}", "KNN": "N/A"}
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
