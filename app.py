from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    name = request.form['name']
    # Convert input into vector
    data = vectorizer.transform([name])

    # Prediction
    prediction = model.predict(data)[0]

    return render_template(
        'index.html',
        prediction_text=f'Predicted Gender: {prediction.upper()}'
    )

if __name__ == '__main__':
    app.run(debug=True)