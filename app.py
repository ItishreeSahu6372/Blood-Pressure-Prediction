import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, render_template

# Flask app with proper static folder
app = Flask(__name__, static_folder="static")

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)
#model = pickle.load(open(r"C:\Users\micky\smtz\flask\model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_bp():
    # Collect form values
    Gender = float(request.form["Gender"])
    Age = float(request.form["Age"])
    History = float(request.form.get("History", 0))          # default 0 if missing
    Patient = float(request.form["Patient"])
    TakeMedication = float(request.form.get("TakeMedication", 0))  # default 0
    Severity = float(request.form["Severity"])
    BreathShortness = float(request.form["BreathShortness"])
    VisualChange = float(request.form["VisualChanges"])
    NoseBleeding = float(request.form["NoseBleeding"])
    Whendiagnoused = float(request.form["Whendiagnoused"])  # exact name from training
    Systolic = float(request.form["Systolic"])
    Diastolic = float(request.form["Diastolic"])
    ControlledDiet = float(request.form["ControlledDiet"])

    # Prepare features in the correct order
    features_values = np.array([[Gender, Age, History, Patient, TakeMedication, Severity,
                                 BreathShortness, VisualChange, NoseBleeding, Whendiagnoused,
                                 Systolic, Diastolic, ControlledDiet]])

    df = pd.DataFrame(features_values, columns=[
        'Gender','Age','History','Patient','TakeMedication','Severity',
        'BreathShortness','VisualChanges','NoseBleeding','Whendiagnoused',
        'Systolic','Diastolic','ControlledDiet'
    ])

    # Make prediction
    prediction = model.predict(df)

    # Map numeric output to readable stage
    if prediction[0] == 0:
        result = "NORMAL"
    elif prediction[0] == 1:
        result = "HYPERTENSION (Stage-1)"
    elif prediction[0] == 2:
        result = "HYPERTENSION (Stage-2)"
    else:
        result = "HYPERTENSIVE CRISIS"

    text = "Your Blood Pressure stage is: "

    # Send prediction + patient input details to result page
    details = {
        "Gender": Gender,
        "Age": Age,
        "History": History,
        "Patient": Patient,
        "TakeMedication": TakeMedication,
        "Severity": Severity,
        "BreathShortness": BreathShortness,
        "VisualChanges": VisualChange,
        "NoseBleeding": NoseBleeding,
        "Whendiagnoused": Whendiagnoused,
        "Systolic": Systolic,
        "Diastolic": Diastolic,
        "ControlledDiet": ControlledDiet
    }

    return render_template("result.html", prediction_text=text + result, details=details)

@app.route('/detail')
def detail():
    return render_template("detail.html")

if __name__ == "__main__":
    app.run(debug=True,port=5000,use_reloader=False)




