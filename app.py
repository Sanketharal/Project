from flask import Flask, request, render_template
import pandas as pd
# from sklearn.linear_model import LinearRegression
import pickle

# create flask app
app = Flask(__name__)

# Load your trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    number_of_servers = int(request.form['number_of_servers'])
    season = int(request.form['season'])
    server_time = int(request.form['server_time'])
    surrounding_temp = float(request.form['surrounding_temp'])  # Update to match HTML input name
    current_temp = float(request.form['current_temp'])
    maintainable_temp = float(request.form['maintainable_temp'])

    # Prepare the input data in the correct format
    input_data = pd.DataFrame([[number_of_servers, season, server_time, surrounding_temp, current_temp, maintainable_temp]],
                              columns=['Number of Servers', 'Season', 'Server Time (hours)', 'Surrounding Temperature (°C)', 'Current Server Temperature (°C)', 'Maintainable Server Temperature (°C)'])

    # Make prediction using the model
    prediction = model.predict(input_data)[0]

    # Render the prediction result
    return render_template('index.html', prediction_text=f'Total Water Required: {prediction:.2f} liters')

if __name__ == "__main__":
    app.run(debug=True)
