import pickle
count_vectorizer = pickle.load(open("models/cv.pkl", 'rb'))
classifier_model = pickle.load(open("models/clf.pkl", 'rb'))

def make_pred(email):
    tokenized_email = count_vectorizer.transform([email]) 
    prediction = classifier_model.predict(tokenized_email)
    if prediction == 1:
        pred = 1
    else:
        pred = -1
    return pred