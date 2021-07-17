<img src="https://user-images.githubusercontent.com/60045850/126037645-ab8f44dd-1060-411e-81ea-80c7e13b79ac.jpeg" alt="DisasterTweets" width="1600" height="450" />


[![Dataset](https://img.shields.io/badge/Dataset-Kaggle-blue)](https://www.kaggle.com/c/nlp-getting-started) [![Language](https://img.shields.io/badge/Lang-Python-brightgreen)](https://www.python.org/) [![Library](https://img.shields.io/badge/Library-Nltk%2C%20Tensorflow-orange)](https://stackshare.io/stackups/nltk-vs-tensorflow) [![ML Library](https://img.shields.io/badge/ML-Scikit--learn-yellowgreen)](https://scikit-learn.org/) [![DL Library](https://img.shields.io/badge/DL-Keras-red)](https://keras.io/)

# Project Description

Twitter has become an important communication channel in times of emergency.   
The ubiquitousness of smartphones enables people to announce an emergency they’re observing in real-time.   
Because of this, more agencies are interested in programatically monitoring Twitter (i.e. disaster relief organizations and news agencies).

- Total approach towards the project can be seen on kaggle

  - **Machine Learning approach** : https://www.kaggle.com/mohitnirgulkar/disaster-tweets-classification-using-ml
  - **Deep Learning approach** : https://www.kaggle.com/mohitnirgulkar/disaster-tweets-classification-using-deep-learning

# Project Contents

1. Exploratory Data Analysis
2. EDA after Data Cleaning 
3. Data Preprocessing using NLP
4. Machine Learning models for classifying Tweets data
5. Deep Learning approach for classifying Tweets data
6. Model Deployment

# Resources Used
- **Packages** : Pandas, Numpy, Matplotlib, Plotly, Word-cloud, Tensorflow, Scikit-Learn, Keras, Keras-tuner, Nltk etc.
- **Dataset**  : https://www.kaggle.com/c/nlp-getting-started
- **Word Embeddings** : https://www.kaggle.com/danielwillgeorge/glove6b100dtxt

## 1. Exploratory Data Analysis
  
  - **Visualising Target Variable of the Dataset**
 
    <img src="https://plotly.com/~raklugrin01/1.png?share_key=hgjA8Zkl35RjZtywNHe0jm" alt="Target Variable" width="1500" height="450" />
 
  - **Visualising Length of Tweets**

    <img src="https://plotly.com/~raklugrin01/3.png?share_key=c65IIAyuBQBfgU1Rfovdfb" alt="Tweet Length" width="1500" height="450" />

  - **Visualising Average word lengths of Tweets**

    <img src="https://plotly.com/~raklugrin01/5.png?share_key=tfNQPMyUblqOh7JL1sEiqW" alt="Avg Word Lengths" width="1500" height="450" />

  - **Visualising most common stop words in the text data**

    <img src="https://plotly.com/~raklugrin01/13.png?share_key=icoxxtajqMGbKIizrTLUX0" alt="Stopwords" width="1500" height="450" />

  - **Visualising most common punctuations in the text data**

    <img src="https://plotly.com/~raklugrin01/15.png?share_key=9JgPThmm677jJmNjJTc0BZ" alt="Punctuations" width="1500" height="450" />

## 2. EDA after Data Cleaning

  - We use Python Regex library and nltk lemmatizing methods for Data Cleaning

  - **Visualising words inside Real Disaster Tweets** 

    <img src="https://www.kaggleusercontent.com/kf/66079405/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..MKc1gh5zr7QGaWXRbdwpYw.fBrplVzbqWWANUi2tKrKUZRJ7eQ0zv7oQHGEoAWqpgGr0PRV0h2FbFpvf7utRHnmEsVbhJ334GtJDw7IdoYeDq73Qwa3bBbwDnXI3CihNyRI-fA-1GcQtDQkS6eob_j_O9TcPbNk1EY9_vhreEoDD-yB1glqOfWYYjibe2LaN-_RKhqhfwQTCiXQFPYNjatrCMqfnLBdUKxOc-rtCcAuA0xVkbXk45QXpQQ15imx_-jVw7aj9WOO7-i9lwZ5C_BzI03RD_NOZQqO41Tyuki7p6pi4sWzsqzZjPu3RH9q8_B-xpb3KTSQaiGgNjcYLMqN5Ta1swz4DZyZrqygU20g4VGynU9TtJNGFs87liiwZPXFyb0v4e2MC-tZihGp4S0cPqzpdX2H-ba11iUkW6KOn17d8LNRreiFTwO5cdOaC1vckSwo0_fv7R1UtSp9St5G469K8F7nhOb4YTkITEyK7GJCV173INnMi1spHA4A5QIFd7_1Jbv_AnZRFHHFx8X92QAiljJIdBk-z5-OWJLTagdOvTW2F57BO3Udyq_rV0u8maqiaO58hxyyEjiRRthlx6xSJsyGdR5TZRtKqP4PKFVF_kbxRqBZj6R9cl0EpLhoq_c6FqqldO3pzD096FrH9pNAvir7xa0vZ2cOUKw146g0wuPlQz_ONN23nP_HpofzmyfXMzBOzteU0nxXKp2At6Km9L6cuO4oSE_tOj3OZg.e_xfRSeCqs_9r153utQF6Q/__results___files/__results___55_0.png" alt="Real WC" width="600" height="400" />


  - **Visualising words inside Fake Disaster Tweets**

    <img src="https://www.kaggleusercontent.com/kf/66079405/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..MKc1gh5zr7QGaWXRbdwpYw.fBrplVzbqWWANUi2tKrKUZRJ7eQ0zv7oQHGEoAWqpgGr0PRV0h2FbFpvf7utRHnmEsVbhJ334GtJDw7IdoYeDq73Qwa3bBbwDnXI3CihNyRI-fA-1GcQtDQkS6eob_j_O9TcPbNk1EY9_vhreEoDD-yB1glqOfWYYjibe2LaN-_RKhqhfwQTCiXQFPYNjatrCMqfnLBdUKxOc-rtCcAuA0xVkbXk45QXpQQ15imx_-jVw7aj9WOO7-i9lwZ5C_BzI03RD_NOZQqO41Tyuki7p6pi4sWzsqzZjPu3RH9q8_B-xpb3KTSQaiGgNjcYLMqN5Ta1swz4DZyZrqygU20g4VGynU9TtJNGFs87liiwZPXFyb0v4e2MC-tZihGp4S0cPqzpdX2H-ba11iUkW6KOn17d8LNRreiFTwO5cdOaC1vckSwo0_fv7R1UtSp9St5G469K8F7nhOb4YTkITEyK7GJCV173INnMi1spHA4A5QIFd7_1Jbv_AnZRFHHFx8X92QAiljJIdBk-z5-OWJLTagdOvTW2F57BO3Udyq_rV0u8maqiaO58hxyyEjiRRthlx6xSJsyGdR5TZRtKqP4PKFVF_kbxRqBZj6R9cl0EpLhoq_c6FqqldO3pzD096FrH9pNAvir7xa0vZ2cOUKw146g0wuPlQz_ONN23nP_HpofzmyfXMzBOzteU0nxXKp2At6Km9L6cuO4oSE_tOj3OZg.e_xfRSeCqs_9r153utQF6Q/__results___files/__results___59_0.png" alt="Fake WC" width="600" height="400" />

  - **Visualising top 10 N-grams where N is 1,2,3**

    <img src="https://plotly.com/~raklugrin01/17.png?share_key=rHBUmASeWITErHR7rEdZqJ" alt="Top N-grams" width="1500" height="1000" />

## 3. Data Preprocessing using NLP

  - Data Preprocessing for ML models is done using two approaches
  
    - **Bag of Words** using CountVectorizer
    - **Term Frequency and Inverse Document Frequency** using TfidfVectorizer

  - Data Preprocessing for DL models using Tokenization 

## 4. Machine Learning models for classifying Tweets data

  - **Machine Learning Models using different n-grams and both Bow and Tf-Idf and visualisation comparing there accuracy** 
  
    <img src="https://user-images.githubusercontent.com/60045850/124084027-31b1ef80-da6c-11eb-9d71-86db60c64927.png" alt="Ml List" width="480" height="330" />

  - **The Best ML model trained as we can see above is Voting Classifer, whose report and confusion matrix is shown below**

    <img src="https://user-images.githubusercontent.com/60045850/124084392-8d7c7880-da6c-11eb-8e49-eacd0766c6df.png" alt="Voting Classifier" width="600" height="520" />

## 5. Deep Learning approach for classifying Tweets data

  - Using **Glove Word Embeddings** of embedding dimension = 100 to get embedding matrix for our DL models
  - For every DL model we create a function and use **Keras-Tuner** to tune the model
  - Finally we choose Bidirectional LSTM for the Deployment

## 6. Model Deployment

  - [Bidirectinal LSTM](https://machinelearningmastery.com/develop-bidirectional-lstm-sequence-classification-python-keras/#:~:text=Bidirectional%20LSTMs%20are%20an%20extension,LSTMs%20on%20the%20input%20sequence.) model obtained from Deep Learning approach is used for deployment
  - Micro Web Framework [Flask](https://flask.palletsprojects.com/) is used to create web app 
  - Heroku is used to deploy the our Web-app on https://disastertweetsdl.herokuapp.com/
  - Deep Learning Web app working

  ![Deployment Demo](https://github.com/raklugrin01/DisasterTweets/blob/main/Readme_requirements/demo.gif)

# Scope of Improvemment

  - We can always use large dataset which covers almost every type of data for both machine learning and deep learning
  - We can use the best pretrained models but they require a lot of computational power
  - Also there are various ways to increase model accuracy like k-fold cross validation, different data preprocessing techniques better than used here

# Conclusion

The Data analysis and modelling was sucessfully done, and the Deep Learning model was deployed on [Heroku](https://disastertweetsdl.herokuapp.com/)

Please do ⭐ the repository, if it helped you in anyway.

