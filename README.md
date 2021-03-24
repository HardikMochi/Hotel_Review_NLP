<h1 align='center'>HOTEL REVIEWS CLASSIFIER</h1>

<p align="center">
  <img src="https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/five_stars.png" width=600>
</p>

<strong> Here is a demo application of the review classifier: https://hotelreviewnlp.herokuapp.com/ </strong> 

Try it out by inputting a review that you found online, or wrote yourself, and it will clasify the review with a score between 1-5.
<table><tr><td><img src='https://github.com/HardikMochi/Hotel_Review_NLP/blob/master/image/hotel.PNG' width=500></td><td><img src='https://github.com/HardikMochi/Hotel_Review_NLP/blob/master/image/negative.PNG' width=500></td></tr></table>

## Business Case
Potential customers, could have their hotel choice be influenced by a tweet. Opinions are shared constantly on social media platforms, and are read by their followers. The knowledge, of what these followers think about our hotel, from reading these online posts, could help us better understand the general public's perception of our hotel. 

By using sentiment analysis, on existing hotel reviews from Tripadvisor.com, I created a model that can quantify on a scale of 1-5, how the author of a tweet on twitter, or a post on a reddit thread, feels about our hotel, and as a result, also how the readers think about us. If a review classifies to be less than a score of 3, this post/tweet could be looked into, find out why they had a negative opinion of our hotel, and in return fix the problem

## Table of Contents
<details open>
<summary>Show/Hide</summary>
<br>


1. [ File Descriptions ](#File_Description)
2. [ Technologies Used ](#Technologies_Used)    
3. [ Structure ](#Structure)
4. [ Executive Summary ](#Executive_Summary)
   * [ 1. simple EDA](#EDA_and_Cleaning)
   * [ 2. Further EDA and Preprocessing ](#Further_EDA_and_Preprocessing) 
   * [ 3. Imbalance ](#Imbalance)
   * [ 4. Modelling and Hyperparameter Tuning ](#Modelling)
   * [ 4. Neural Network Modelling ](#Neural_Network_Modelling)
   * [ 6. Revaluation and Deployment ](#Revaluation)
</details>

## File Descriptions
<details>
<a name="File_Description"></a>
<summary>Show/Hide</summary>
<br>
    
* <strong>[ Data ](https://github.com/HardikMochi/Hotel_Review_NLP/tree/master/data)</strong>: folder containing all data files
   
* <strong>[ Images ](https://github.com/HardikMochi/Hotel_Review_NLP/tree/master/image)</strong>: folder containing images used for README and presentation pdf
* <strong>[ Models ](https://github.com/HardikMochi/Hotel_Review_NLP/tree/master/Models)</strong>: folder containing trained models saved with pickle
    * <strong> Decision Tree.pkl, KNN.pkl, Logistic Regression.pkl, Neural Network.pkl, Random Forest.pkl, Stacking.pkl, SVM.pkl, XGBoost.pkl</strong>

* <strong>[ 1.0 Simple EDA.ipynb ](https://github.com/HardikMochi/Hotel_Review_NLP/blob/master/1.0%20Simple%20EDA.ipynb)</strong>: notebook with early data exploration and data manipulation
* <strong>[ 2.0 EDA_and_Preprocessing.ipynb ](https://github.com/HardikMochi/Hotel_Review_NLP/blob/master/2.0%20EDA_and_Preprocessing.ipynb)</strong>: notebook with feature engineering and nlp preprocessing
* <strong>[ 3.0 Imbalance.ipynb ](https://github.com/HardikMochi/Hotel_Review_NLP/blob/master/3.0%20Imbalance.ipynb)</strong>: transform imbalance to balance data
* <strong>[ 4.0 Modelling_and_Hyperparameter_Tuning.ipynb ](https://github.com/HardikMochi/Hotel_Review_NLP/blob/master/4.0%20Modelling_and_Hyperparameter_Tuning.ipynb)</strong>:  notebook with all the models created
* <strong>[ 5.Neural_Network_Modelling ](https://github.com/HardikMochi/Hotel_Review_NLP/blob/master/5.0%20Neural_Network_Modelling.ipynb)</strong>: notebook with  model training using neural networks to get better result and comparing old best model and NN model, and deployment
* <strong>[ Classification.py ](https://github.com/HardikMochi/Hotel_Review_NLP/blob/master/Classification.py)</strong>: contains classes with classifcation methods
* <strong>[ Ensemble.py ](https://github.com/HardikMochi/Hotel_Review_NLP/blob/master/Ensemble.py)</strong>: contains classes with ensemble methods
* <strong>[ App.py ](https://github.com/HardikMochi/Hotel_Review_NLP/blob/master/app.py)</strong>: contains script to run app
* <strong>[ Procfile ](https://github.com/HardikMochi/Hotel_Review_NLP/blob/master/Procfile)</strong>: file supporting Heroku application
* <strong>[ requirements.txt ](https://github.com/HardikMochi/Hotel_Review_NLP/blob/master/requirements.txt)</strong>: dependencies for heroku application
</details>

## Tecnologies Used:
<details>
<a name="Technologies_Used"></a>
<summary>Show/Hide</summary>
<br>
    
* <strong>Python</strong>
* <strong>Pandas</strong>
* <strong>Numpy</strong>
* <strong>Matplotlib</strong>
* <strong>Seaborn</strong>
* <strong>NLTK</strong>
* <strong>Scrapy</strong>
* <strong>Scikit-Learn</strong>
* <strong>Keras</strong>
* <strong>Tensorflow</strong>
* <strong>Streamlit</strong>
* <strong>Heroku</strong>
</details>

   
<a name="Executive_Summary"></a>
## Executive Summary


<a name="=EDA_and_Cleaning"></a>
### Early EDA, and Cleaning:
<details open>
<summary>Show/Hide</summary>
<br>
 data is not perfectly balance  as you see in this below picture
 <h5 align="center">Histogram of Scores for Each Hotel</h5>
<p align="center">
  <img src="https://github.com/HardikMochi/Hotel_Review_NLP/blob/master/image/2.PNG" width=600>
</p>
<a name="EDA_and_Preprocessing"></a>

### Further EDA and Preprocessing
<details open>
<summary>Show/Hide</summary>
<br>
 I started with some analysis on the text columns; review and review summary.

Using the FreqDist function in the ntlk library I plotted a graph with the most frequent words and phrases in both columns. Stopwords were removed to capture the more meaningful words.

<h5 align="center">Distribution Plot of Frequent Words and Phrases in Text ( Review Summary (Left) and Review (Right) )</h5>
<table><tr><td><img src='https://github.com/HardikMochi/Hotel_Review_NLP/blob/master/image/3.PNG' width=500></td><td><img src='https://github.com/HardikMochi/Hotel_Review_NLP/blob/master/image/4.PNG' width=500></td></tr></table>

I had noticed a lot of the most frequent words in the review text happened to be words with no sentimental impact, so I iteratively removed unmeaningful words such as 'room', 'hotel', 'hilton' etc. I did this as a precaution, as some of these words may impact my model accuracies.

<h5 align="center">World Cloud of Frequent Words and Phrases in Text After Removing Unmeaningful Words ( Review Summary (Left) and Review (Right) )</h5>
<table><tr><td><img src='https://github.com/HardikMochi/Hotel_Review_NLP/blob/master/image/5.PNG' width=500></td><td><img src='https://github.com/HardikMochi/Hotel_Review_NLP/blob/master/image/6.PNG' width=500></td></tr></table>

after that feature words I applied stemmation and lemmitisation to both the reviews and review summaries. 

<h5 align="center">Example of Lemmatisation and Stemmation Applied to a Review and Review Summary</h5>
<p align="center">
  <img src="https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/lemm_stemm_ex.png" width=600>
</p>

Stemmation had broken down some words into words that don't exist, whereas lemmitisation had simplified adjectives and verbs to their root form. I chose to continue with the lemmitised version of the texts for further processing.

Prior to vectorising the current dataset, I did a train, test split to save the test data for after modelling.

Using the lemmed texts for review and review summary I used TF-IDF vectorisation with an ngram range of 2, leaving me with a vectorised dataset with 138 words and phrases (112 from reviews and 26 from review summaries). I then saved the x and y train data in separate csv files for modelling.
</details>

<a name="Imbalance"></a>
### Data Imbalance:
<details open>
<summary>Show/Hide</summary>
<br>
Data set was imbalance so I tryied diffrent diffrent methods to improve the result.
<br>So for that i created one base line model to check the result. and the result that i got is in below picture.

<p align="center">
  <img src="https://github.com/HardikMochi/Hotel_Review_NLP/blob/master/image/7.PNG" width=600>
</p>


<a name="Modelling"></a>
### Modelling:
<details open>
<summary>Show/Hide</summary>
<br>

I have created .py files; Classifiction.py and Ensemble.py with classes, that contain functions to simplify the modelling process, and to neaten up the modelling notebook.

I did another data split into Train and Validation data in preparation for using GridSearch Cross Validation. I also chose Stratified 5-fold has a my choice for cross validating.

For the majority of models I created, I applied hyperparameter tuning, where I started with a broad range of hyperparameters, and tuned for optimal train accuracy and validation accuracy. 


<h5 align="center">Table Comparing Best Models</h5>
<p align="center">
  <img src="https://github.com/HardikMochi/Hotel_Review_NLP/blob/master/image/8.PNG" width=600>
</p>

Initially, I thought the validation accuracy was low for most of the models I created, but when considering these models were to classify for 5 different classes, 0.45 is also good (where 0.2 = randomly guessing correctly).

I have saved all the models using the pickle library's dump function and stored them in the Models folder.
</details>


<a name="Neural_Network_Modelling"></a>
### Neural Network Modelling:
<details open>
<summary>Show/Hide</summary>
<br>
    
I experimented with different classifcation and ensemble methods to help classify hotel review scores. so to Imporove the resultI wanted to explore a deep learning approach. 
    
    
<h5 align="center">Neural Network Architecture</h5>
<p align="center">
  <img src="https://github.com/awesomeahi95/Hotel_Review_NLP/blob/master/Images/NN_architecture.png" width=600>
</p>
    
* Input Layer: 17317 Nodes (one for each word in training data + 4 extra for padding, unknown words, start of review, and unused words)
* Embedding Layer: takes 17317 unique items and maps them into a 16 dimensional vector space
* Global Average 1D Pooling Layer: scales down 16 dimensional layer
* Dense Hidden Layer: 16 Nodes (using relu activation function)
* Dense Output Layer: 5 nodes for each score (using sigmoid activation function)
    
</details>

<a name="Revaluation"></a>
### Revaluation and Deployment:
<details open>
<summary>Show/Hide</summary>
<br>

I tested the neural network model using the test data and achieved an accuracy of <strong>0.5710</strong> which is better than the SVM model accuracy of <strong>0.51</strong>, by <strong>over 5%</strong>. 
    
I wanted to look at the confusion matrix, as this gives a better idea of how the model is performing over all 5 classes.
    
<h5 align="center">Neural Network Model Test Confusion Matrix</h5>
<p align="center">
  <img src="https://github.com/HardikMochi/Hotel_Review_NLP/blob/master/image/9.PNG" width=600>
</p>

#### Deployment and Application
    
    
I planned on future improvements being the addition of the neural network model and then creating an application for the model, so as a next step I decided to make a working application to test out new reviews using streamlit. I have deployed the app using Heroku:  https://hotelreviewnlp.herokuapp.com/. 
    
Using this model, we will learn more about our new and old customers, then we can improve Hilton Hotel's guest satisfaction, and as a result increase customer retention and bring in new travelers.
    

    
</details>
