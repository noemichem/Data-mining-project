#pip instaal flask should be installed before running this code

#run the code on terminal with python app.py
#open the browser and type http://127.0.0.1:5000/


from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load(r'C:\Users\LENOVO\Documents\GitHub\Data-mining-project\notebooks_final\Algorithms\random_forest_model.pkl')


# A mapping for region codes (example, adjust to your mapping)
regions = { 0: 'Africa',
            1: 'East Asia',
            2: 'Eastern Europe',
            3: 'Latin America',
            4: 'Middle East',
            5: 'North America',
            6: 'Northern Europe',
            7: 'Oceania',
            8: 'South Asia',
            9: 'Southern Europe',
            10: 'Western Europe'}

@app.route('/')
def home():
    return render_template('index.html', regions=regions)

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from form
        region = int(request.form['region'])
        danceability = float(request.form['danceability'])
        energy = float(request.form['energy'])
        key = int(request.form['key'])
        loudness = float(request.form['loudness'])
        mode = int(request.form['mode'])
        speechiness = float(request.form['speechiness'])
        acousticness = float(request.form['acousticness'])
        instrumentalness = float(request.form['instrumentalness'])
        liveness = float(request.form['liveness'])
        valence = float(request.form['valence'])
        tempo = float(request.form['tempo'])
        duration_ms = int(request.form['duration_ms'])
        time_signature = int(request.form['time_signature'])

        # Prepare the feature vector for prediction
        features = np.array([[region, danceability, energy, key, loudness, mode, speechiness, acousticness, 
                              instrumentalness, liveness, valence, tempo, duration_ms, time_signature]])

        # Make prediction using your model
        prediction = model.predict(features)

        # Return the result
        if prediction[0] == 1:
            result = "Popular"
        else:
            result = "Not Popular"
        
        return render_template('index.html', result=result, regions=regions)

if __name__ == '__main__':
    app.run(debug=True)
