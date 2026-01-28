from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load kidney model ONCE
model = pickle.load(open("models/kidney.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/kidney", methods=["GET"])
def kidneyPage():
    return render_template("kidney.html")

@app.route("/predict", methods=["POST"])
def predictPage():
    try:
        # Get form data in correct order
        features = [
            float(request.form["age"]),
            float(request.form["bp"]),
            float(request.form["al"]),
            float(request.form["su"]),
            float(request.form["rbc"]),
            float(request.form["pc"]),
            float(request.form["pcc"]),
            float(request.form["ba"]),
            float(request.form["bgr"]),
            float(request.form["bu"]),
            float(request.form["sc"]),
            float(request.form["pot"]),
            float(request.form["wc"]),
            float(request.form["htn"]),
            float(request.form["dm"]),
            float(request.form["cad"]),
            float(request.form["pe"]),
            float(request.form["ane"])
        ]

        features = np.array(features).reshape(1, -1)
        prediction = model.predict(features)[0]

        return render_template("predict.html", pred=int(prediction))

    except Exception as e:
        return render_template(
            "home.html",
            message="⚠️ Please enter valid numerical values for all fields"
        )

if __name__ == "__main__":
    app.run(debug=True)
