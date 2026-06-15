import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string

app = Flask(__name__)

# 1. Load your trained DecisionTreeClassifier model
MODEL_PATH = "cancer.pkl"
try:
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)
except Exception as e:
    print(f"Error loading {MODEL_PATH}: {e}")
    model = None

# 2. Modern HTML UI Layout with integrated CSS styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cancer Survival Prediction Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #090d16 0%, #111827 100%);
            --card-bg: rgba(17, 24, 39, 0.7);
            --card-border: rgba(255, 255, 255, 0.08);
            --input-bg: rgba(31, 41, 55, 0.5);
            --accent-glow: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --success-bg: rgba(16, 185, 129, 0.15);
            --success-text: #34d399;
            --success-border: rgba(16, 185, 129, 0.3);
            --danger-bg: rgba(239, 68, 68, 0.15);
            --danger-text: #f87171;
            --danger-border: rgba(239, 68, 68, 0.3);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: var(--bg-gradient);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
        }

        .wrapper {
            width: 100%;
            max-width: 800px;
            background: var(--card-bg);
            backdrop-filter: blur(20px);
            border: 1px solid var(--card-border);
            border-radius: 24px;
            padding: 2.5rem;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        header h1 {
            font-size: 2.25rem;
            font-weight: 700;
            letter-spacing: -0.05em;
            margin-bottom: 0.5rem;
            background: linear-gradient(to right, #a5b4fc, #e9d5ff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        header p {
            color: var(--text-secondary);
            font-size: 1rem;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem;
        }

        @media (max-width: 640px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 0.5rem;
        }

        label {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-secondary);
            letter-spacing: 0.02em;
        }

        input, select {
            background: var(--input-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 0.85rem 1.2rem;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.95rem;
            transition: all 0.2s ease;
            outline: none;
        }

        input:focus, select:focus {
            border-color: #818cf8;
            box-shadow: 0 0 0 4px rgba(129, 140, 248, 0.15);
        }

        .btn-submit {
            grid-column: span 2;
            background: var(--accent-glow);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 1.1rem;
            font-size: 1rem;
            font-weight: 700;
            cursor: pointer;
            transition: transform 0.1s ease, box-shadow 0.2s ease;
            margin-top: 1rem;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.2);
        }

        @media (max-width: 640px) {
            .btn-submit {
                grid-column: span 1;
            }
        }

        .btn-submit:hover {
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
            transform: translateY(-1px);
        }

        .btn-submit:active {
            transform: translateY(1px);
        }

        .result-container {
            margin-top: 2.5rem;
            padding: 1.75rem;
            border-radius: 16px;
            text-align: center;
            animation: slideUp 0.4s cubic-bezier(0.16, 1, 0.3, 1);
            border: 1px solid transparent;
        }

        .result-container.Alive {
            background: var(--success-bg);
            border-color: var(--success-border);
            color: var(--success-text);
        }

        .result-container.Deceased {
            background: var(--danger-bg);
            border-color: var(--danger-border);
            color: var(--danger-text);
        }

        .result-title {
            font-size: 0.875rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 700;
            margin-bottom: 0.5rem;
            opacity: 0.8;
        }

        .result-value {
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: -0.03em;
        }

        @keyframes slideUp {
            from { opacity: 0; transform: translateY(16px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>

<div class="wrapper">
    <header>
        <h1>Oncology Predictive Analytics</h1>
        <p>Enter clinical metrics below to compute model survival prognosis</p>
    </header>

    <form method="POST" action="/predict">
        <div class="form-grid">
            <div class="form-group">
                <label for="Age">Age</label>
                <input type="number" id="Age" name="Age" min="0" max="120" value="45" required>
            </div>

            <div class="form-group">
                <label for="Gender">Gender</label>
                <select id="Gender" name="Gender" required>
                    <option value="0">Female (0)</option>
                    <option value="1" selected>Male (1)</option>
                </select>
            </div>

            <div class="form-group">
                <label for="State">State (Encoded Integer)</label>
                <input type="number" id="State" name="State" min="0" value="1" required>
            </div>

            <div class="form-group">
                <label for="City">City (Encoded Integer)</label>
                <input type="number" id="City" name="City" min="0" value="1" required>
            </div>

            <div class="form-group">
                <label for="Cancer_Type">Cancer Type (Encoded Integer)</label>
                <input type="number" id="Cancer_Type" name="Cancer_Type" min="0" value="2" required>
            </div>

            <div class="form-group">
                <label for="Stage">Cancer Stage (Numerical)</label>
                <input type="number" id="Stage" name="Stage" min="1" max="4" value="2" required>
            </div>

            <div class="form-group">
                <label for="Treatment_Type">Treatment Type (Encoded Integer)</label>
                <input type="number" id="Treatment_Type" name="Treatment_Type" min="0" value="0" required>
            </div>

            <div class="form-group">
                <label for="Survival_Months">Survival Months</label>
                <input type="number" id="Survival_Months" name="Survival_Months" min="0" value="12" required>
            </div>

            <button type="submit" class="btn-submit">Run Model Diagnostics</button>
        </div>
    </form>

    {% if prediction %}
    <div class="result-container {{ prediction }}">
        <div class="result-title">Prognostic Output Prediction</div>
        <div class="result-value">{{ prediction }}</div>
    </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE, prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, prediction="Error: Model missing.")
        
    try:
        input_data = {
            "Age": float(request.form.get("Age")),
            "Gender": float(request.form.get("Gender")),
            "State": float(request.form.get("State")),
            "City": float(request.form.get("City")),
            "Cancer_Type": float(request.form.get("Cancer_Type")),
            "Stage": float(request.form.get("Stage")),
            "Treatment_Type": float(request.form.get("Treatment_Type")),
            "Survival_Months": float(request.form.get("Survival_Months"))
        }
        
        features_df = pd.DataFrame([input_data])
        prediction_result = model.predict(features_df)[0]
        output_text = str(prediction_result)
        
    except Exception as error:
        output_text = f"Processing Error: {str(error)}"

    return render_template_string(HTML_TEMPLATE, prediction=output_text)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
