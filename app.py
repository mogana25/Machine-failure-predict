import pickle
import numpy as np
from flask import Flask, request, render_template
from flask_mysqldb import MySQL

# Create Flask app
app = Flask(__name__)

# MySQL configuration
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'machine_predict'

# Initialize MySQL
mysql = MySQL(app)

# Load all models
models = {
    "random_forest": pickle.load(open("model.pkl", "rb")),
    "logistic_regression": pickle.load(open("logistic_model.pkl", "rb")),
    "svm": pickle.load(open("svm_model.pkl", "rb")),
    "decision_tree": pickle.load(open("decision_tree_model.pkl", "rb")),
}

@app.route("/")
def Home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Get selected model
    model_choice = request.form.get('model_choice')
    model = models.get(model_choice)

    if not model:
        return render_template("index.html", prediction_text="Invalid model selected!")

    # Extract features
    feature_names = ["footfall", "tempMode", "AQ", "USS", "CS", "VOC", "RP", "IP", "Temperature"]
    float_features = [float(request.form.get(name, 0)) for name in feature_names]

    # Perform prediction
    prediction = model.predict([np.array(float_features)])
    fail = int(prediction[0])  # Assuming 0 or 1 prediction

    # Raw input for DB storage
    footfall = request.form['footfall']
    tempMode = request.form['tempMode']
    AQ = request.form['AQ']
    USS = request.form['USS']
    CS = request.form['CS']
    VOC = request.form['VOC']
    RP = request.form['RP']
    IP = request.form['IP']
    Temperature = request.form['Temperature']

    # Insert prediction result into MySQL
    cursor = mysql.connection.cursor()
    query = """
        INSERT INTO machine (
            footfall, tempMode, AQ, USS, CS, VOC, RP, IP, Temperature, fail, model_choice
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    cursor.execute(query, (
        footfall, tempMode, AQ, USS, CS, VOC, RP, IP, Temperature, fail, model_choice
    ))
    mysql.connection.commit()
    cursor.close()

    return render_template("index.html", prediction_text=f"Prediction: {prediction[0]}")

@app.route('/view')
def view():
    cursor = mysql.connection.cursor()
    query = "SELECT * FROM machine"
    cursor.execute(query)
    predictions = cursor.fetchall()
    cursor.close()
    return render_template("view.html", a=predictions)

if __name__ == "__main__":
    app.run(debug=True)








