# Flask utils
from flask import Flask, request, render_template
import numpy as np
import pickle
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import nltk
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
app = Flask(__name__)

model = load_model('model_BiLSTM.h5')
# loading
with open('tokenizer.pickle', 'rb') as handle:
    word_tokenizer = pickle.load(handle)


def model_predict(text):

    lemmatizer = WordNetLemmatizer()
    #removal of url
    
    text = re.sub(r'https?://\S+|www\.\S+|http?://\S+',' ',text[0]) 
    
    #decontraction
    text = re.sub(r"won\'t", " will not", text)
    text = re.sub(r"won\'t've", " will not have", text)
    text = re.sub(r"can\'t", " can not", text)
    text = re.sub(r"don\'t", " do not", text)    
    text = re.sub(r"can\'t've", " can not have", text)
    text = re.sub(r"ma\'am", " madam", text)
    text = re.sub(r"let\'s", " let us", text)
    text = re.sub(r"ain\'t", " am not", text)
    text = re.sub(r"shan\'t", " shall not", text)
    text = re.sub(r"sha\n't", " shall not", text)
    text = re.sub(r"o\'clock", " of the clock", text)
    text = re.sub(r"y\'all", " you all", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"n\'t've", " not have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'d've", " would have", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ll've", " will have", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    text = re.sub(r"\'re", " are", text)
    
    #removal of html tags
    text = re.sub(r'<.*?>',' ',text) 
    
    # Match all digits in the string and replace them by empty string
    text = re.sub(r'[0-9]', '', text)
    text = re.sub("["
                           u"\U0001F600-\U0001F64F"  # removal of emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+",' ',text)
    
    # filtering out miscellaneous text.
    text = re.sub('[^a-zA-Z]',' ',text) 
    text = re.sub(r"\([^()]*\)", "", text)
    
    # remove mentions
    text = re.sub('@\S+', '', text)  
    
    # remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', text)  
    

    # Lowering all the words in text
    text = text.lower()
    text = text.split()


    text = [lemmatizer.lemmatize(words) for words in text if words not in stopwords.words('english')]
    
    # Removal of words with length<2
    text = [i for i in text if len(i)>2] 
    text = ' '.join(text)
    
    common_words = ['via','like','build','get','would','one','two','feel',
                'lol','fuck','take','way','may','first','latest','want',
                'make','back','see','know','let','look','come','got',
                'still','say','think','great','pleas','amp']

    text = [' '.join(i for i in text.split() if i not in common_words)]

    text = word_tokenizer.texts_to_sequences(text)

    # 24 is length of longest train sentence

    text = pad_sequences(text,23,padding='post')
    
    preds = model.predict(text)
    print(preds)
    pred = (preds[0][0] > 0.42).astype(np.int)
    
    return pred


@app.route('/', methods=['Get'])
def index():
    #home page
    return render_template('home.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == 'POST':
        tweet = request.form['tweet']
        data = [tweet]
        my_pred = model_predict(data)
        return render_template('result.html',prediction = my_pred)

@app.route('/home', methods=['Get'])
def home():
    #home page
    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=False)
