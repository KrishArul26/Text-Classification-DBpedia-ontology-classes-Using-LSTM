# 


<h2 align="center">  Text-Classification-Using-LSTM</h2>

<h3 align="left">Hierarchical Taxonomy of Wikipedia article classes Classification-Using-LSTM </h3>

 <p align="center">
  <img width="600" src="https://user-images.githubusercontent.com/74568334/140572521-72125b9d-69c1-442e-9b74-ef60ce6a8b2e.png">
</p> 

<h3 align="left">Introduction </h3>

 
<p style= 'text-align: justify;'> Text classification is the task of assigning a set of predefined categories to free text. Text classifiers can be used to organize, structure, and categorize pretty much anything. For example, new articles can be organized by topics, support tickets can be organized by urgency, chat conversations can be organized by language, brand mentions can be organized by sentiment, and so on.</p>


<p align="center">
  <img width="800" src="https://user-images.githubusercontent.com/74568334/140572780-58814fa5-52aa-4b52-bd1c-cfa70dc0ba65.jpeg">
</p> 

<h2 align="center"> Technologies Used </h2>
 
 ```
 1. IDE - Pycharm
 2. LSTM - As a classification Deep learning Model
 3. GPU - P-4000
 4. Google Colab - Text Analysis
 5. Flas- Fast API
 6. Postman - API Tester
 7. Gensim - Word2Vec embeddings
 
 ```
 
<p style= 'text-align: justify;'> 
 
   🔑 Prerequisites
      All the dependencies and required libraries are included in the file requirements.txt

      Python 3.6
 
</p>

<h2 align="center"> Dataset </h2>

<p style= 'text-align: justify;'> The DBpedia ontology classification dataset is constructed by picking 14 non-overlapping classes from DBpedia 2014. They are listed in classes.txt. From each of thse 14 ontology classes, we randomly choose 40,000 training samples and 5,000 testing samples. Therefore, the total size of the training dataset is 560,000 and testing dataset 70,000. The files train.csv and test.csv contain all the training samples as comma-sparated values. There are 3 columns in them, corresponding to class index (1 to 14), title and content. The title and content are escaped using double quotes ("), and any internal double quote is escaped by 2 double quotes (""). There are no new lines in title or content. </p>

For Dataset Please click [here](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/version/1)


<h2 align="center"> Implementations </h2>

<h4 align="left"> In this section, contains the project directory, explanation of each python file presents in the directory.  </h2>


<h3 align="left">1. Project Directory</h3>


<h4 align="left"> Below picture illustrate the complete folder structure of this project.</h4>


<p align="center">
  <img width="400" src="https://user-images.githubusercontent.com/74568334/140577934-92f60e0d-c905-478e-be62-638bd6a7ad82.png">
</p> 


<h3 align="left">2. preprocess.py </h3>

<p style= 'text-align: justify;'> Below picture illustrate the preprocess.py file, It does the necessary text cleaning process such as removing punctuation, numbers, lemmatization. And it will create train_preprocessed, validation_preprocessed and test_preprocessed pickle files for the further analysis.</p>

<p align="center">
  <img width="400" src="https://user-images.githubusercontent.com/74568334/140578710-2b346932-32c8-4f60-b9bf-b79fbb4fbf10.png">
</p> 

<h3 align="left">3. word_embedder_gensim.py </h3>

<p style= 'text-align: justify;'> Below picture illustrate the word_embedder_gensim.py, After done with text pre-processing, this file will take those cleaned text as input and will be creating the Word2vec embedding for each word.</p>

<p align="center">
  <img width="400" src="https://user-images.githubusercontent.com/74568334/140579065-79a7e215-1f8f-4715-816c-0247d007a520.png">
</p> 


<h3 align="left">4. rnn_w2v.py </h3>

<p style= 'text-align: justify;'>Below picture illustrate the rnn_w2v.py, After done with creating Word2vec for each word then those vectors will use as input for creating the LSTM model and Train the LSTM (RNN) model with body and Classes. </p>

<p align="center">
  <img width="400" src="https://user-images.githubusercontent.com/74568334/140579999-d0ae2ac4-74bc-460d-82eb-3ee7cbb40a73.png">
</p> 

<h3 align="left">5. main.py </h3>

<p style= 'text-align: justify; Below picture illustrate the main.py, After evaluating the LSTM model, This files will create the Rest -IP, To that It will use FLASK frameworks and get the request from the customer or client then It will Post into the prediction files and Answer will be deliver over the web browser. </p>
 
<p align="center">
  <img width="400" src="https://user-images.githubusercontent.com/74568334/140581040-86b02b9a-fb8c-4f10-9ebf-03e05573f7a6.png">
</p> 



























