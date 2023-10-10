from flask import Flask, render_template, request, jsonify
from utils import make_pred

app = Flask(__name__)

@app.route('/')
def home():
    # text = ""
    # if request.method == 'POST':
    #     text = request.form.get('email-content')
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_api():
    data = request.get_json(force=True) 
    email_text = data['email-content']
    prediction = make_pred(email_text)
    return jsonify({'prediction': prediction, 'email': email_text})  

@app.route("/predict", methods=["POST"])
def predict():
    email_text = request.form.get('email-content')
    prediction = make_pred(email_text)
    return render_template("index.html", prediction=prediction, email=email_text)


if __name__ == '__main__':
    app.run(debug=True)