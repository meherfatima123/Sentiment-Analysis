from flask import Flask,render_template,request
from pickle import load
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

stemmer = PorterStemmer()

def preprocess_test(raw_review):
    # Remove Special Character
    letters_only = re.sub('[^a-zA-Z]',' ',raw_review)
    
    # Conver sentence into Lower Case
    letters_only = letters_only.lower()
    
    # Tokenize
    words = letters_only.split()
    
    #Remove Stop Words
    words = [w for w in words if not w in stopwords.words('english')]
    
    # Stemming
    
    words = [stemmer.stem(word) for word in words]
    
    clean_review = ' '.join(words)
    
    return clean_review 

def predict(review):
    
    vectorizer = load(open('model/vect_full.pkl','rb'))
    model = load(open('model/amzfood_logistic_full.pkl','rb'))
    
    clean_review = preprocess_test(review)
    
    clean_review_vector = vectorizer.transform([clean_review])
    
    prediction = model.predict(clean_review_vector)
    
    return prediction








app=Flask(__name__)

@app.route('/')
def sentiment():
    return render_template('sentiment.html')

@app.route('/res',methods=['POST'])
def display():
    a=request.form.get('in_1')
    prediction=predict(a)
 
    if prediction == 0:
        rev='Negative Sentiment'
    else:
        rev='Positive Sentiment'

    print(prediction)

    return render_template('display.html',r=a,rev=rev)
    

if __name__=='__main__':
    app.run(debug=True)