from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)
cv = pickle.load(open("models/cv.pkl", 'rb'))
clf = pickle.load(open("models/clf.pkl", 'rb'))

@app.route('/')
def home():
    # text = ""
    # if request.method == 'POST':
    #     text = request.form.get('email-content')
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_api():
    data = request.get_json(force=True) 

    email = data['content']
    tokenized_email = cv.transform([email]) # X 
    prediction = clf.predict(tokenized_email)
    # If the email is spam prediction should be 1
    prediction = 1 if prediction == 1 else -1
    
    return jsonify({'prediction': prediction, 'email': email})  # Return 

@app.route("/predict", methods=["POST"])
def predict():
    email = request.form.get('content')
    tokenized_email = cv.transform([email]) # X 
    prediction = clf.predict(tokenized_email)
    prediction = 1 if prediction == 1 else -1
    return render_template("index.html", prediction=prediction, email=email)


if __name__ == '__main__':
    # app.run(host='0.0.0.0', debug=True)
    app.run(debug=True)