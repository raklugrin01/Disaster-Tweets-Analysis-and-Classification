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

    <img src="https://www.kaggleusercontent.com/kf/65636581/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..l7SXjoTM0ZM4QOObMo4qQg.YrCQtgdBeesRCXIH6Wt8ze4ntHZV8cOxSz_tWa8HEzg-y29tH549cPn2yeoAicbpqazuolDvLMFJcVvfpzn7w7heI4hfs_6yTzE1GUwjnoB3HTZswM-qHWlaTmDjm-TLLeLFx0LlLWaR0GAs2eu3kyBVaAsBeAFpLTbuSVxliAxTF2-iJ2ADDm-CHTZKH2iwvqFrCvbq3dZWOefrLkZrI6bmRNXlOZItVv3kjTYoayQ0Qw3gJzc0-2KQ6QGVJndeWrZN4HzkHgwfhfW9RqywjbE_y1jzygwvjpQ9a06tfUXxnkxGB1nEiXJd785AY4A8oP_e4CLLWE7CanhZtuzfU4l34zCDsF2M6KObeTl9QLoR46BPTaZLs-3QS2ndbWBQzwTqpzbxMbG8BRLd_5TkVkLbgc-2KllxYHIVmyIVwBWXAEL-AXwTEDWnDpQKhrURLItpZRZbHa7OUNGGVGGrrVi0VlDLr0CoABiOgJWMi0vAlkGgrNXS2fF-mKPVx1zqUgcBcFJ0MIVbP_a3k3Kjmfeu-wVhuNhmueqk8ZOZGbQ7sJhSHdCM3JtpUyNmnGZaFIlUYBTqtcqfTUPvYIjFBOh7OE9BOVVnt4sJa6IDC9WQRtLI7u0e-TeNFOq-bZ6KIQMAwpewGtCZ-_uzDZ9L7vR8lheqSDGTZP_4swOVZRJ3FeiwLaK-9Z38bvhlG2Jy.ooPYUjwwtQpoxDUCcMgm6A/__results___files/__results___54_0.png" alt="Real WC" width="600" height="400" />


  - **Visualising words inside Fake Disaster Tweets**

    <img src="https://www.kaggleusercontent.com/kf/65636581/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..l7SXjoTM0ZM4QOObMo4qQg.YrCQtgdBeesRCXIH6Wt8ze4ntHZV8cOxSz_tWa8HEzg-y29tH549cPn2yeoAicbpqazuolDvLMFJcVvfpzn7w7heI4hfs_6yTzE1GUwjnoB3HTZswM-qHWlaTmDjm-TLLeLFx0LlLWaR0GAs2eu3kyBVaAsBeAFpLTbuSVxliAxTF2-iJ2ADDm-CHTZKH2iwvqFrCvbq3dZWOefrLkZrI6bmRNXlOZItVv3kjTYoayQ0Qw3gJzc0-2KQ6QGVJndeWrZN4HzkHgwfhfW9RqywjbE_y1jzygwvjpQ9a06tfUXxnkxGB1nEiXJd785AY4A8oP_e4CLLWE7CanhZtuzfU4l34zCDsF2M6KObeTl9QLoR46BPTaZLs-3QS2ndbWBQzwTqpzbxMbG8BRLd_5TkVkLbgc-2KllxYHIVmyIVwBWXAEL-AXwTEDWnDpQKhrURLItpZRZbHa7OUNGGVGGrrVi0VlDLr0CoABiOgJWMi0vAlkGgrNXS2fF-mKPVx1zqUgcBcFJ0MIVbP_a3k3Kjmfeu-wVhuNhmueqk8ZOZGbQ7sJhSHdCM3JtpUyNmnGZaFIlUYBTqtcqfTUPvYIjFBOh7OE9BOVVnt4sJa6IDC9WQRtLI7u0e-TeNFOq-bZ6KIQMAwpewGtCZ-_uzDZ9L7vR8lheqSDGTZP_4swOVZRJ3FeiwLaK-9Z38bvhlG2Jy.ooPYUjwwtQpoxDUCcMgm6A/__results___files/__results___57_0.png" alt="Fake WC" width="600" height="400" />

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

