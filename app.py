from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

# Load Models & Tools
rf_model = joblib.load("models/rf_model.pkl")
xgb_model = joblib.load("models/xgb_model.pkl")
inc_model = joblib.load("models/inc_model.pkl")
scaler = joblib.load("models/scaler.pkl")
encoder = joblib.load("models/encoder.pkl")
columns = joblib.load("models/columns.pkl")
accuracies = joblib.load("models/accuracies.pkl")

@app.route("/")
def home():
    # Print encoder classes to debug
    print("Encoder classes:")
    for col in encoder:
        print(f"{col}: {encoder[col].classes_}")
    return render_template("index.html", columns=columns, encoder=encoder)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Debug input values
        print("Form inputs:")
        for col in columns:
            print(f"{col}: {request.form.get(col)}")
            
        data_row = []

        for col in columns:
            value = request.form.get(col)
            if not value:
                return jsonify({'error': f"Missing input for {col}"})

            # Debugging for internetservice
            if col == 'internetservice':
                print(f"Internet service value: {value}")
                print(f"Available options: {encoder[col].classes_}")
                if value not in encoder[col].classes_:
                    print(f"WARNING: {value} not in encoder classes for {col}")
                    # Try to find a close match
                    if value.lower() == 'dsl':
                        for option in encoder[col].classes_:
                            if option.lower() == 'dsl':
                                value = option
                                print(f"Matched to {value}")
                                break

            if col in encoder:
                if value not in encoder[col].classes_:
                    return jsonify({'error': f"Invalid value '{value}' for {col}. Valid options are: {list(encoder[col].classes_)}"})
                value = encoder[col].transform([value])[0]
            else:
                value = float(value)

            data_row.append(value)

        data_row = np.array(data_row).reshape(1, -1)
        scaled_input = scaler.transform(data_row)

        # Get predictions from all models
        rf_pred = rf_model.predict(scaled_input)[0]
        rf_conf = rf_model.predict_proba(scaled_input)[0][1] * 100
        
        xgb_pred = xgb_model.predict(scaled_input)[0]
        xgb_conf = xgb_model.predict_proba(scaled_input)[0][1] * 100
        
        inc_pred = inc_model.predict(scaled_input)[0]
        inc_conf = inc_model.predict_proba(scaled_input)[0][1] * 100

        # Process churn prediction result
        if rf_pred == 1:
            churn_result = "Churn"
            churn_message = "Customer is likely to churn. Consider retention strategies."
        else:
            churn_result = "Not Churn"
            churn_message = "Customer is likely to stay. Consider loyalty programs."

        # Create response with all model predictions and confidences
        return jsonify({
            'result': churn_result,
            'message': churn_message,
            'rf_pred': int(rf_pred),
            'xgb_pred': int(xgb_pred),
            'inc_pred': int(inc_pred),
            'rf_acc': float(round(accuracies['Random Forest'] * 100, 2)),
            'xgb_acc': float(round(accuracies['XGBoost'] * 100, 2)),
            'inc_acc': float(round(accuracies['InceptionNet'] * 100, 2)),
            'rf_conf': float(round(rf_conf, 2)),
            'xgb_conf': float(round(xgb_conf, 2)),
            'inc_conf': float(round(inc_conf, 2))
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': f"Error: {str(e)}"})

@app.route("/upload", methods=["POST"])
def upload():
    try:
        file = request.files['file']
        if not file.filename.endswith('.csv'):
            return render_template("index.html", columns=columns, encoder=encoder, error="Invalid File Format")

        df = pd.read_csv(file)

        os.makedirs("static", exist_ok=True)

        charts = [
            ('churn_dist.png', sns.countplot, {'x': 'churnvalue'}),
            ('tenure_churn.png', sns.boxplot, {'x': 'churnvalue', 'y': 'tenuremonths'}),
            ('charges_churn.png', sns.boxplot, {'x': 'churnvalue', 'y': 'monthlycharges'}),
            ('contract_churn.png', sns.countplot, {'x': 'contract', 'hue': 'churnvalue'}),
            ('internet_churn.png', sns.countplot, {'x': 'internetservice', 'hue': 'churnvalue'}),
        ]

        for filename, plot_func, args in charts:
            plt.figure()
            plot_func(data=df, **args)
            plt.title(filename.split('.')[0].replace('_', ' ').title())
            plt.xticks(rotation=30)
            plt.tight_layout()
            plt.savefig(f'static/{filename}')
            plt.close()

        return render_template("index.html", columns=columns, encoder=encoder, charts=True)

    except Exception as e:
        return render_template("index.html", columns=columns, encoder=encoder, error=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)