#!/usr/bin/env python
# coding: utf-8

# In[296]:


import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import re

from sklearn.preprocessing import MinMaxScaler
def print_validation_report(y_true, y_pred):
    print("Classification Report")
    print(classification_report(y_true, y_pred))
    acc_sc = accuracy_score(y_true, y_pred)
    print("Accuracy : "+ str(acc_sc))
    
    return acc_sc
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import wordcloud

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize 

import os


# In[297]:


def print_validation_report(y_true, y_pred):
    print("Classification Report")
    print(classification_report(y_true, y_pred))
    acc_sc = accuracy_score(y_true, y_pred)
    print("Accuracy : "+ str(acc_sc))
    
    return acc_sc


# In[298]:


def plot_confusion_matrix(y_true, y_pred):
    mtx = confusion_matrix(y_true, y_pred)
    #fig, ax = plt.subplots(figsize=(4,4))
    sns.heatmap(mtx, annot=True, fmt='d', linewidths=.5,  
                cmap="Blues", cbar=False, ax=ax)
    #  square=True,
    plt.ylabel('true label')
    plt.xlabel('predicted label')


# In[299]:


data = pd.read_csv("Desktop/essays.csv",encoding='utf-8')
data=data.dropna()
data["essay_text"]=data["essay_text"].str.lower()
data.head()


# In[419]:


depara={}
for id,item in enumerate(data.theme_title.value_counts().index.tolist()):
    depara[item]=id
depara


# In[301]:


data['tipo_redacao'] = data['theme_title'].map(depara).astype(int)


# In[302]:


data['length'] = data['essay_text'].str.len().astype(int)


# In[303]:


from nltk.corpus import stopwords
import string
string.punctuation
stopwords=stopwords.words("portuguese")


# In[304]:


def remove_punctuation_and_stopwords(text):
    no_punctuation =  ""
    for ch in text:
        if ch in string.punctuation:
            ch = " "
        no_punctuation +=ch
    no_punctuation = "".join(no_punctuation).split()
    
    no_punctuation_no_stopwords =         " ".join([word for word in no_punctuation if word not in stopwords])
    return re.sub(' +', ' ', no_punctuation_no_stopwords)
remove_punctuation_and_stopwords("testando aqui um novo teste-de teste..haha")


# In[305]:


from nltk.stem.snowball import SnowballStemmer

Stemmer=SnowballStemmer("portuguese")
Stemmer.stem("teremos")


# In[306]:


def sinonimos(palavra,level=4):
    return list(dict.fromkeys([j for i in [[lem.name().lower() for lem in syn.lemmas("por")[0:level]] for syn in wn.synsets(palavra,lang="por")[0:level]] for j in i]))
sinonimos("vida")


# In[402]:


def replaceSinonimos(text):
    out=""
    utilizados={}
    for temp_palavra in text.split(" "):
        loc=0
        for fim in ["","a","e","i","o","u","s"]:
            palavra=temp_palavra+fim
            for item in utilizados:
                for sin in [j for i in utilizados[item] for j in i]:
                    if sin == palavra and item != sin:
                        out+=""+item + " "
                        loc=1
                    elif sin == palavra and item == sin:
                        loc=1
                        out+=""+temp_palavra + " "
            if loc == 0:
                try:
                    utilizados[temp_palavra].append(sinonimos(palavra))
                except:
                    utilizados[temp_palavra] = sinonimos(palavra)
        if loc == 0:
            out+=temp_palavra + " "
    print("done")
    return out
        
replaceSinonimos("recent vida recente frescor") 


# In[387]:


lista= [[],['fresco', 'recente', 'novo'], [], [], [], [], [], [], ['fresco', 'recente', 'novo'], [], [], [], []]
[j for i in lista for j in i]


# In[398]:


replaceSinonimos(data["text"][2])


# In[399]:


data["count_stopwords"]=data.apply(lambda col : len([word for word in col.essay_text.split() if word in stopwords]),axis=1)
data["biggest_stopwords"]=data.apply(lambda col : len(max([word for word in col.essay_text.split() if word in stopwords],key=len)),axis=1)
data["essay_text2"]=data['essay_text'].apply(remove_punctuation_and_stopwords)


# In[400]:


data["essay_text3"]=data['essay_text2'].apply(replaceSinonimos)


# In[401]:


data["text2"]=data.apply(lambda col : " ".join([Stemmer.stem(word) for word in col.essay_text3.split()]),axis=1)


# In[403]:


data["text"]=data['text2'].apply(replaceSinonimos)


# In[ ]:


data["essay_text3"][2]


# In[ ]:





# In[254]:


#data=data.drop(["id","essay_text","essay_title","theme_text","theme_title","score_1","score_2","score_3","score_4","score_5","total_score"],axis=1)


# In[255]:


data


# In[405]:


from sklearn.feature_extraction.text import CountVectorizer
bow_transformer = CountVectorizer().fit(data['text'])
print(len(bow_transformer.vocabulary_))


# In[406]:


bow_data = bow_transformer.transform(data['text'])


# In[407]:


print( bow_data.nnz / (bow_data.shape[0] * bow_data.shape[1]) *100 )


# In[408]:


from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(bow_data)


# In[409]:


data_tfidf = tfidf_transformer.transform(bow_data)
type(data_tfidf)


# In[410]:


data_treino=data_tfidf.A
data_treino2=np.append(data_treino, data["length"].values.reshape(-1,1), axis=1)
data_treino3=np.append(data_treino2, data["biggest_stopwords"].values.reshape(-1,1), axis=1)
data_treino4=np.append(data_treino3, data["count_stopwords"].values.reshape(-1,1), axis=1)
data_treino4


# In[441]:


from sklearn.model_selection import train_test_split

data_tfidf_train, data_tfidf_test, label_train, label_test =     train_test_split(data_treino4, data["tipo_redacao"], test_size=0.40, random_state=50)


# In[442]:


import lightgbm as lgb
params = {

    'task' : 'train',
    'boosting_type' : 'gbdt',
    'objective' : 'multiclass',
    'metric' : ['multi_error'],
    'min_data_in_bin':1,
    'num_leaves' : 60,
    'learning_rate' : 0.05,
    'verbose' : 0,
    'num_class' : 35
}
lgb_train = lgb.Dataset(data_tfidf_train.data, label_train)
lgb_eval = lgb.Dataset(data_tfidf_test.data, label_test, reference=lgb_train)
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=4000,
                valid_sets=lgb_eval,
                early_stopping_rounds=200)
#out=gbm.predict(x_test)


# In[444]:


out=gbm.predict(data_tfidf_test)
output = pd.DataFrame({'real': label_test, 'pred': np.argmax(out, axis=1)})
output.loc[output.real == output.pred, 'acertou'] = 'True' 
output.loc[output.real != output.pred, 'acertou'] = 'False' 
output.groupby("acertou").count().apply(lambda x:
                                                 100 * x / float(x.sum()))


# In[445]:


from sklearn.metrics import confusion_matrix
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
confusion_matrix(label_test, np.argmax(out, axis=1))


# In[429]:





# In[464]:


i=[0,1]
y_train=label_train.apply(lambda x: 1 if x in i else 0)
y_test=label_test.apply(lambda x: 1 if x in i else 0)
y_test.groupby(y_test).count()


# In[454]:


import lightgbm as lgb
params = {

    'task' : 'train',
    'boosting_type' : 'gbdt',
    'objective' : 'binary',
    'metric' : ['auc'],
    'min_data_in_bin':1,
    'num_leaves' : 10,
    'learning_rate' : 0.05,
    'verbose' : 0
}
lgb_train = lgb.Dataset(data_tfidf_train.data, y_train)
lgb_eval = lgb.Dataset(data_tfidf_test.data, y_test, reference=lgb_train)
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=4000,
                valid_sets=lgb_eval,
                early_stopping_rounds=200)


# In[463]:


np.count_nonzero(np.around(gbm.predict(data_tfidf_test.data),5) > 0.2)


# In[467]:


gbm.(data_tfidf_test.data)


# In[479]:


#Import modules
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc
sns.set_style("whitegrid")

#Define 'x' sets
train_x = np.asarray(data_tfidf_train.data)
test_x = np.asarray(data_tfidf_test.data)
train_y=np.asarray(y_train.data)
test_y=np.asarray(y_test.data)
#------------------------Build LightGBM Model-----------------------
train_data=lgb.Dataset(train_x, label=train_y)

#Select Hyper-Parameters
params = {'boosting_type': 'gbdt',
          'max_depth' : -1,
          'objective': 'binary',
          'nthread': 5,
          'num_leaves': 64,
          'learning_rate': 0.07,
          'max_bin': 512,
          'subsample_for_bin': 200,
          'subsample': 1,
          'subsample_freq': 1,
          'colsample_bytree': 0.8,
          'reg_alpha': 1.2,
          'reg_lambda': 1.2,
          'min_split_gain': 0.5,
          'min_child_weight': 1,
          'min_child_samples': 5,
          'scale_pos_weight': 1,
          'num_class' : 1,
          'metric' : 'binary_error'
          }

# Create parameters to search
gridParams = {
    'learning_rate': [0.07],
    'n_estimators': [8,16],
    'num_leaves': [20, 24, 27],
    'boosting_type' : ['gbdt'],
    'objective' : ['binary'],
    'random_state' : [501], 
    'colsample_bytree' : [0.64, 0.65],
    'subsample' : [0.7,0.75],
    #'reg_alpha' : [1, 1.2],
    #'reg_lambda' : [ 1.2, 1.4],
    }

# Create classifier to use
mdl = lgb.LGBMClassifier(boosting_type= 'gbdt',
          objective = 'binary',
          n_jobs = 5, 
          silent = True,
          max_depth = params['max_depth'],
          max_bin = params['max_bin'],
          subsample_for_bin = params['subsample_for_bin'],
          subsample = params['subsample'],
          subsample_freq = params['subsample_freq'],
          min_split_gain = params['min_split_gain'],
          min_child_weight = params['min_child_weight'],
          min_child_samples = params['min_child_samples'],
          scale_pos_weight = params['scale_pos_weight'])

# View the default model params:
mdl.get_params().keys()

# Create the grid
grid = GridSearchCV(mdl, gridParams, verbose=2, cv=4, n_jobs=-1)

# Run the grid
grid.fit(train_x, train_y)

# Print the best parameters found
print(grid.best_params_)
print(grid.best_score_)

# Using parameters already set above, replace in the best from the grid search
params['colsample_bytree'] = grid.best_params_['colsample_bytree']
params['learning_rate'] = grid.best_params_['learning_rate']
# params['max_bin'] = grid.best_params_['max_bin']
params['num_leaves'] = grid.best_params_['num_leaves']
#params['reg_alpha'] = grid.best_params_['reg_alpha']
#params['reg_lambda'] = grid.best_params_['reg_lambda']
params['subsample'] = grid.best_params_['subsample']
# params['subsample_for_bin'] = grid.best_params_['subsample_for_bin']

print('Fitting with params: ')
print(params)

#Train model on selected parameters and number of iterations
lgbm = lgb.train(params,
                 train_data,
                 280,
                 #early_stopping_rounds= 40,
                 verbose_eval= 4
                 )

#Predict on test set
predictions_lgbm_prob = lgbm.predict(test_x)
predictions_lgbm_01 = np.where(predictions_lgbm_prob > 0.5, 1, 0) #Turn probability to 0-1 binary output

#--------------------------Print accuracy measures and variable importances----------------------
#Plot Variable Importances
lgb.plot_importance(lgbm, max_num_features=21, importance_type='split')

#Print accuracy
acc_lgbm = accuracy_score(test_y,predictions_lgbm_01)
print('Overall accuracy of Light GBM model:', acc_lgbm)

#Print Area Under Curve
plt.figure()
false_positive_rate, recall, thresholds = roc_curve(test_y, predictions_lgbm_prob)
roc_auc = auc(false_positive_rate, recall)
plt.title('Receiver Operating Characteristic (ROC)')
plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out (1-Specificity)')
plt.show()

print('AUC score:', roc_auc)

#Print Confusion Matrix
plt.figure()
cm = confusion_matrix(test_y, predictions_lgbm_01)
labels = ['No Default', 'Default']
plt.figure(figsize=(8,6))
sns.heatmap(cm, xticklabels = labels, yticklabels = labels, annot = True, fmt='d', cmap="Blues", vmin = 0.2);
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()


# In[ ]:




