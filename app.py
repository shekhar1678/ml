import subprocess
from sklearn.model_selection import train_test_split
import pickle
import joblib
import time 
from sklearn.metrics.pairwise import cosine_similarity
import numpy as  np
import os
import time
from sklearn.metrics.pairwise import euclidean_distances
from nltk.corpus import stopwords
from sklearn.preprocessing import MinMaxScaler
import re
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from flask import Flask, request , jsonify
import flask
app = Flask(__name__)

@app.route('/')
def index():
    return flask.render_template('index.html')

def create_glove_embedding(X_train_title,column_name='pre_title'):
  preprocessed_title = X_train_title[column_name].values
  tfidf_model = TfidfVectorizer(min_df=10,ngram_range=(1,4), max_features=300)
  tfidf_model.fit(preprocessed_title)
  dictionary = dict(zip(tfidf_model.get_feature_names(), list(tfidf_model.idf_)))
  tfidf_words = set(tfidf_model.get_feature_names())
  return dictionary,tfidf_words,preprocessed_title

def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase


def remove_tags(string):
    # https://developpaper.com/analyze-the-meaning-of-in-regular-expression/
    # dot means anything can go here and * means zero or more
    # a.*?bMatches the shortest string that starts with a and ends with B. If you apply it toaababIt will matchaab(first to third characters) andab(fourth to fifth characters). 
    result = re.sub('<.*?>','',string)
    return result

def remove_all(words):
    # convert everything else other than  alphabet characters
    # inside square bracket it is used as a not operator
    # outside of the bracket it is used to match the first character
    words = words.lower() 
    result = re.sub('[^a-zc++c#]',' ',words)
    result = result.lower()
    return result

def remove_words(word):
    # remove all the stop words
    neword = [i for i in word.split() if i not in nltk.corpus.stopwords.words('english')]
    neword = ' '.join(neword)
    return neword

def execute(command):
        pipe = subprocess.Popen(command,shell=True,universal_newlines=True,
               stderr=subprocess.PIPE,stdout=subprocess.PIPE)
        output,error = pipe.communicate()
        print (output)
        print ("error  = ",error)
               
def start():
    # downloading the prprocessed csv file 
    if os.path.exists('preprocessed_data.csv') == False:
        execute ('gdown --id 1zLs4DTIhmnHzLUHZyR7ZMoEZxxH7AAB4')
        # storing it in the pandas datframe
    if os.path.exists('glove_vectors') == False:
        execute ('gdown --id  1lDca_ge-GYO0iQ6_XDLWePQFMdAA2b8f')
    
    return df1


def preprocess(string):
    nltk.download('stopwords')
    string = decontracted(string)
    string = remove_tags(string)
    string = remove_all(string)
    string = remove_words(string)
    return string

# decorator to calculate total time function is taking

def wrapper(function):
  '''
  This is used as decorator to calculate the total time taken by 
  the function.
  '''
  def callable(*args,**kwargs):
    start = time.time()
    b = function(*args,**kwargs)
    end = time.time()
    print ("<====> total time taken is {} seconds <====>".format(end-start))
    return b
  
  return callable  


def res_complete(all_vectors,single_vectors,n,df1):
    '''
    This is helper function to  create all the metrics we need 
    like cosine, eculidean and the score value of the question tables.
    '''
    # find the cosine values
    cosine,cvalues = find_similarity(all_vectors,single_vectors,n,df1,complete=True)
    # finc the inverse euclidean distance
    inverse,values = find_similarity(all_vectors,np.array(single_vectors),n,df1,cosine=False,complete=True) 
    df_glove = pd.DataFrame()
    # create a dataframe and store the result of the cosine and inverse 
    # eucludian distance seperately
    df_glove['question'] = list(cosine.keys())
    df_glove['cosine_value'] = list(cosine.values())
    df_glove['score'] = [df1.iloc[i]['score'] for i  in cvalues]
    # normalizing the score values
    df_glove['score'] =  MinMaxScaler().fit_transform(np.array(df_glove['score']).reshape(-1,1))
    df_glove1  = pd.DataFrame()
    df_glove1['question'] = list(cosine.keys())
    df_glove1['euc'] = list(cosine.values())
    # normalizing the euclidean distance as the values can be greater than 0 and 1
    # normailizing it so that the values stays between 0 and 1
    df_glove1['euc'] = MinMaxScaler().fit_transform(np.array(df_glove1['euc']).reshape(-1,1))
    df_glove1['score'] = [df1.iloc[i]['score'] for i  in cvalues]
    df_glove1['score'] =  MinMaxScaler().fit_transform(np.array(df_glove1['score']).reshape(-1,1))
    return df_glove,df_glove1

@wrapper
def glove_vectors(df1,glove_words,model_glove,text=None,n=10,complete=False):
  '''
  This is a generic function to calculate glove_vectors and the cosine similarity
  if parameter complete is true we calculate all the metrics score, like cosine 
  and euclidean and normalized score of the column score of the table.
  '''
  dictionary,tfidf_words,preprocessed_title = create_glove_embedding(df1)
  tfidf_w2v_vectors = create_average_tfidf_w2v(preprocessed_title,tfidf_words,dictionary,glove_words,model_glove)
  
  ##############################################################################
  # infernece 
  vector = np.zeros(300)
  vector = create_average_tfidf_w2v([text],tfidf_words,dictionary,glove_words,model_glove)
  print (" ====   To find the similar query for  ====")
  print (text)    
  print ("============================================")
  ##############################################################################
  # finding only cosine similarity
  if complete==True:
    return res_complete(tfidf_w2v_vectors,vector,n,df1)
  

def total_score(func_name,pre_text,df1,glove_words,model_glove,n=10):
  '''
  This function computes the total score with formulae 0.6*cosine_score+0.2*
  inveted euclidena_score + 0.2*score_of_question  
  '''
  cosine_weight = 0.6
  euc_weight = 0.2
  score_weight = 0.2  

  df_ans,df_ans1 = func_name(df1,glove_words,model_glove,text=pre_text,complete=True,n=10)
  # https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html
  df3 = pd.merge(df_ans,df_ans1)
  # fill the values of nan with zero
  df3.fillna(0)
  final_score = []
  # iterate the dataframe and use the formulae
  for index,row in df3.iterrows():
    final_score.append((row['cosine_value']*0.6) + (row['score']*0.2) + (row['euc']*0.2))

  df3['final_score']=final_score
  df3=df3.sort_values(by=['final_score'],ascending=False)
  return df3

def find_similarity(vectors,vector,n,df1,cosine=True,complete=False):
    '''
    This function is used to find the cosine similarity 
    Input : 
         vectors : The vector represenation of complete dataset used in training
         the data.
         vector : The vector represnetation of single query vector for which we
         want to find the similar vector.
         n : top n number of vector which are similar to the query vector
    '''
    if cosine == True:
        # reshape done because ValueError: Expected 2D array, got 1D array instead
        result =  cosine_similarity((vector[0]).reshape(1, -1), Y=vectors, dense_output=True)
    else:
        result =  euclidean_distances((vector).reshape(1, -1), Y=vectors)
    # output will be list of list with only one element 
    dot_result = result[0]
    # we want indices that can sort the array
    # we than reverse it as it will give us sorted indices in reverse order
    # we find the top n number of them
    values = list(reversed((np.argsort(dot_result))))[:n]
    dict_1 = { df1.iloc[i]['title']:dot_result[i] for i in values}
    # as mp.queue is process and thread safe
    if complete==True:
      return dict_1,values  
    else:
      return dict_1

def create_average_tfidf_w2v(preprocessed_title,tfidf_words,dictionary,glove_words,model_glove):
  '''
  This method calculate the average tfidf of the 
  sentece we find the find the word of a sentence if present in both glove and 
  tfidf then calculate its weighted tfidf using pretrained glove vectors.
  '''
  # average Word2Vec
  # compute average word2vec for each review.
  print ("first")
  tfidf_w2v_vectors = []; # the avg-w2v for each sentence/review is stored in this list
  for sentence in preprocessed_title: # for each review/sentence
    vector = np.zeros(300) # as word vectors are of zero length
    tf_idf_weight =0; # num of words with a valid vector in the sentence/review
    for word in sentence.split(): # for each word in a review/sentence
        if (word in glove_words) and (word in tfidf_words):
            vec = model_glove[word] # getting the vector for each word
            # here we are multiplying idf value(dictionary[word]) and the tf value((sentence.count(word)/len(sentence.split())))
            tf_idf = dictionary[word]*(sentence.count(word)/len(sentence.split())) # getting the tfidf value for each word          
            vector += (vec * tf_idf) # calculating tfidf weighted w2v
            tf_idf_weight += tf_idf
    if tf_idf_weight != 0:
        vector /= tf_idf_weight
    tfidf_w2v_vectors.append(vector)

  print("len of tfidf vector = ",len(tfidf_w2v_vectors))
  print("len of individual vector=",len(tfidf_w2v_vectors[0]))
  return tfidf_w2v_vectors

@app.route('/result', methods=['POST'])
def app_name():
    df1 = start()
    # TO DO remove this when things get complete
    print ("enter a input")
    output = request.form.to_dict()
    out = output['var1']
    # got the preprocessed input from the string 
    string = preprocess(out)
    with open('glove_vectors', 'rb') as f:
        model_glove = pickle.load(f)
        glove_words =  set(model_glove.keys())
    X_train_title, X_test = train_test_split(df1, test_size=0.2)
    ans = total_score(glove_vectors,string,X_train_title,glove_words,model_glove)
    print (ans['question'])
    return flask.render_template('result.html',ans=list(ans['question'].values))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)