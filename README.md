# Clothing-data-NLP-Analysis-and-Recommender

A course project of Analyzing Unstructured Data class at UCSD

## 1. Description

In this project, we conduct analysis on the rating and fit of customers based on the e-commerce clothing data collected from RentTheRunway.com, and then develop a recommender to return similar items given the user inputs. From the model results, we conclude that review texts could largely improve the performance of classifiers which only includes customer demographics and body measurements. Other than that, word embedding models such as Skip-gram and CBOW which focus on learning from context perform best among a series of text mining models.


## 2. Data Preprocessing

In this project, we use open-sourced clothing fit data which is collected from *RentTheRunway*. The *RentTheRunway* dataset contains 192,544 records from 105,508 users regarding 5,850 items. Our original source of data is in json format. To better manipulate data and conduct further analysis, we convert it to a pandas dataframe. As Figure 1 shows, the raw dataset contains missing values and some of its body measurement indexes are not in numeric formats.



![image-20230313185424732](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/202303132206113.png)

Figure 1: First 5 rows of raw dataset



To handle these issues, first of all, we check the proportion of missing values in each column.  we can see that 6 features contain missing values. For these columns, we fill in their modes to replace missing values. And for feature columns which we expect to have numeric, we transform them to their corresponding units. For bust size and weight, we extract the number part as feature values, and create a new variable cup to store the letters after the number in bust size. And for height, since the original values are in feet and inches, we extract the numbers of feet and inches, convert 1 feet to 12 inches, and recalculate these
values in inches to obtain the height data in numeric format.

![image-20230313221641775](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/202303132216624.png)

Figure 2: Proportion of missing values

From Figure 2, we can see that 6 features contain missing values. For these columns, we fill in their modes to replace missing values. And for feature columns which we expect to have numeric, we transform them to their corresponding units. For *bust size* and *weight*, we extract the number part as feature values, and create a new variable cup to store the letters after the number in *bust size*. And for *height*, since the original values are in feet and inches, we extract the numbers of feet and inches, convert 1 feet to 12 inches, and recalculate these values in inches to obtain the height data in numeric format.

And for text variables, we use **nltk** package to conduct tokenization and stemming operations for them, then we convert tokens to lower cases and remove the stop words. Specifically, we find the list of stop words in **nltk.corpus** package does not contain some frequent stop words in review text, so we manually added about 30 words into this list to keep it consistent with our data.



## 3. Exploratory Data Analysis

To explore the characteristics of each column, we conduct univariate analysis for features and response
variables.

For our 4 response variables fit, rating, rented for, and category, since rating only have 5 levels of response value and the other 3 are all categorical variables, we create some barplot to explore their distributions.

![png](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/202303132206280.png)

Figure 3: Proportion of variables

    



From Figure 3 we find that for fit, more than 70% of customers think the clothes that they get fits all, and
the probabilities of being small or being large are about 10% for each level.

For *rating*, 10 (the highest rating) has the highest proportion of about 60%, and as rating decreases, the proportion also decreases. These 2 variables indicate that most customers are satisfied with the products
and services offered by *RentTheRunway*. 

And from the distribution of variables *rented for* and *category*, we find that most popular categories are
formal dresses which are required for some public affairs. It makes sense since *RentTheRunway* offers
clothes sharing services. For daily dressing which has higher re-use probability, people are more likely
to make purchases instead of renting. 

![png](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/202303132206983.png)

Figure 4: Proportion of categorical features

         

Then for features, we use barplot for categorical features and boxplot for numeric features. In Figure
4, we can see that most customers have medium body shapes. It may be because these kinds of
people could find more suitable clothes on a cloth-sharing website and the website also has a
willingness to offer more clothes in medium sizes since they will have higher turnover rate and
generate more profit.

![png](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/202303132206513.png)

Figure 5: Distribution of numeric features

    

In Figure 5, we use boxplot to explore and examine the distribution of numeric features. Specifically,
what we should pay attention to is the distribution of *age*. For variable age, we find there are many
outliers and the range of *age* is [0,120], which is obviously abnormal. We assume that some
customers are unwilling to offer their actual ages and inputs below 15 or above 70 may be fake ages. We
replace these values with mode to make the variable more likely to obey a convincing distribution.

Besides univariate analysis, we also generate pairplot for bivariate analysis.

![png](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/202303132206016.png)

    

Figure 6: Pairplot of numeric features

From Figure 6, we find that among 4 features bust_number, weight, height, and age, variable weight shows some positive correlations with height and bust_number from the scatter plot. To further examine the correlations, we build a correlation matrix of these variables. In the correlation matrix, all pairs of variables show positive correlations,  which is consistent with the actual fact that people who are taller or have larger bust numbers will have higher weights. However, the correlation between bust_number and weight is  noteworthy since it is rather high. In our modelling part, we will keep track of these 2 variables to check if we should do some adjustments. 

Then, we perform the VIF test on all variables as a  whole, and the obtained VIF value is 2.28, indicating
that there is no significant multicollinearity among the variables. Therefore, we keep all the current features.

![png](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/202303132206618.png)

   Figure 7: Correlation matrix

         

The *RentTheRunway* dataset also contains 2 text variables, *review_summary* and *review_text*.
Previewing the raw data, we find *review_summary* contains phrases extracted from review_text which
include some key information such as cloth category and customer sentiments, but other customer
features are omitted. Since we need text to generate word vectors and build our model, we choose
*review_text* as our text variable since it could provide more comprehensive information. After
tokenization, we build a word cloud to demonstrate the frequent words in our text.



![png](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/202303132206467.png)

 Figure 8: Word cloud of review_text

        

From our word cloud, “dress” is the most mentioned word, followed by words related to customer
experience such as “fit” and “size”. Among these words, while “fit” is mentioned the most, “little” is
also frequently brought up, which might be an indication of less satisfied customers. Also there are
some frequent positive words like “great”, “perfect” and “loved”, which is consistent with the previous
barplot that customers overall feedback in rating are rather satisfactory.



## 4. Research Problem

There are three goals of our research. First, we want to know whether adding review would increase
model accuracy when predicting whether a clothes fit a person or not. Second, whether adding reviews
variable would increase model performance when predicting customer rating. Third, we build a
recommendation model which recommends clothing categories based on user input.

#### 4.1 Input variables

For predictive models which predict fit and rating, the input variables include bust number, weight,
height, size, age, cup, body type, review. For the recommender, the input is the item which a user rated
the highest in the past. For the reviews in predictive models, we try different preprocessing methods to convert text.  These include converting reviews to vectors applying word2vec (Skip-gram & CBOW), GloVe,
and Tf-Idf.

#### 4.2 Output variables

The output variables for predictive models are fit and rating. For the recommender, the output variable is
item_id, and its corresponding category and rent for, which means that the model would recommend top 5
items which have the highest similarity with the user input.

#### 4.3 Baseline Model

The goal for our project is to see whether adding “text” would improve model accuracy, thus the
baseline model is adding all variables except review. The variables include bust number, weight, height,
size, age, cup, body type. The reason that the model adding text could outperforms the baseline is that
first, we only keep important words in each row, and deleted pronouns, prepositions, and other common
words, such as “I, you, he, she, when, where, what, and, also, to, oh, of”, thus reduced the noise caused by not related words. Second, language models learn patterns from corpus, thus they are able to increase
characteristics for each data entry.

#### 4.4 Methods to compare and evaluate model

For fit predicting model and rating predicting model, we compare the performance of model using
different NLP techniques including ***word2vec*** (Skip-gram & CBOW), ***GloVe***, and ***Tf-Idf.*** To evaluate model performance, we compare *AUC-score*, *F1-score*, and *accuracy* of each model. The higher the AUC, the better the performance of the model at distinguishing classes.



## 5. Proposed Solution

Since our purpose is to classify variables *fit* and *rating*, we try logistic regression without and with
review text to clarify whether text is useful in this task. 

To compare the effect of text on the model, we try different bags of words and word embedding
methods to process review text. Bag of words can build a vocabulary from a corpus of documents and
counts how many times the words appear in each document. To put it another way, each word in the
vocabulary becomes a feature and a document is represented by a vector with the same length of the
vocabulary. This approach causes a significant dimensionality problem: the more documents you
have the larger is the vocabulary, so the feature matrix will be a huge sparse matrix. Therefore, in
order to use this way more efficiently, we do a lot of important preprocessing like word cleaning, stop
word removal, stemming to reduce the dimensionality problem. 

Word embeddings are methods which look for a latent space to preserve the semantics as much as
possible. Words from the vocabulary are mapped to vectors of real numbers. These vectors are calculated
from the probability distribution for each word appearing before or after another. These methods are  able to capture many linguistic regularities of words as well as taking their context into consideration. 

To be more specific, we try ***binary***, ***frequency***, ***TF-IDF***, ***word to vector***, ***GloVe*** and ***CBOW*** to process data and retrain the classification models. Then we compare the models’ accuracy, auroc_rate, F1-score(both macro and micro）and choose the best one as the with-text result. Then we compare it with the model’s result without text and to find the improvement of adding text. 

Finally, to make our project more useful, we design a recommender system based on existing information. The recommendation system is constructed based on the ratings given to the item by existing users and the Jaccard similarity. When existing users enter their id, they can see the top 10 products recommended by the system based on their ratings. In addition, we also output the categories of these ten products and the occurrences used, so that customers can choose according to their own needs.



## 6 Experimental Results

#### 6.1 Prediction Model Comparison 

To reduce the cost of training and tuning, we randomly shuffle the original dataset and pick 50,000
records as our dataset for model comparison. Among these 50,000 records, we split them into a training set with 40,000 records and a testing set with 10,000 records. Table 1 shows the evaluation scores of different fitting models on the testing set. 

![image-20230313203630901](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/202303132206841.png)

As we can see from Table 1, all the 4 models show improvement from baseline, which indicates that
review texts do provide more latent information for us to predict user fit. The performance of these
models is not very different. However, in this project, we believe that AUROC can better reflect the pros
and cons of the model in a balanced manner, so we finally choose the AUROC indicator to select the
model. And among the 4 text models, ***CBOW*** has the best AUROC score.

![image-20230313203735503](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/202303132206126.png)

And from Table 2, when predicting rating, text models also outperform the baseline model. Among
these models, Skip Gram has the best AUROC score. 

From above results, in general, ***word2v*** (CBOW and Skip Gram) has the best performance. Then, because
the model effect of ***CBOW*** and ***skip-gram*** is not very different, and in this task, whether the clothes fit the
customer is more important, finally we chose the ***CBOW*** model. We re-estimate this model on the
whole dataset with 192,544 records and get the following results.

![image-20230313203841462](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/202303132206442.png)

As Table 3, the performances of ***CBOW*** model are similar on partial datasets as well as on the full
dataset, which means our model is stable. 



#### 6.2 Recommender System Examples

In our recommender, based on cosine similarity, for  each user within the dataset, we find 10 most similar
items for them and return the categories and rented_for reasons as well. In Table 4, we randomly
choose a user and show the recommendation for them.

![image-20230313204354035](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/202303132206070.png)



## 7 Conclusion and Outlook

In conclusion, we have the following findings from our analysis.

First, from the data collected from *RentTheRunway*, the majority of its customers have high evaluation of
the services and products. The main reason that customers would like to use cloth sharing service is to prepare for some formal affairs, and most of the customers are of medium figure, which may be because this kind of customers are exactly whom *RentTheRunway* wants to target.

Second, baseline models of logistic regression which excluded text variables show poor performance in
predicting user fit and rating. In Particular, the macro F1 scores of baseline models are extremely low,
which suggests that the classifiers are affected by imbalanced data and have low accuracy in predicting
the minority levels.

Third, ***Skip-gram*** and ***CBOW*** models show the best performance among all the text models involved in
this project. Possible reason is that under this e-commerce situation, a user’s fit and overall evaluation are much more likely to be mentioned in user comments, so these models which generate word embeddings based on the study of context outperform other models.

Also, our project still has some weaknesses which suggest the potential space for improvement.

First, when examining the impact of text compared with other features, we use only *logistic regression*
models. Though models with text features added do perform better than *logistic regression*, if we parse
original features into more complex machine learning models, they may generate better performance. In the future, we will consider comparing text models with other models to further study their impacts.

Secondly, when building text models, we only develop some simple frequency-based models and word embedding models. During the experiment, we tried to build a Bert model, however it failed to return expected outputs in time since it had higher requirements for devices. In the future, if possible, we are going to try more language models with advanced devices.

And for the recommender system we build, it returns recommendations based on existing records of certain users. However it does not solve the cold start problem that we cannot apply it to generate
recommendations for new customers. To address this issue, we have proposed plans including introducing
random numbers, returning average, or asking for more features from new customers that we can use
***KNN*** to find some similar users in our records and then apply our recommender system to them.

And lastly, for the deliverable, we are planning to create some more applicable forms to better demonstrate our models and let them be accessible to amateurs as well. In the future, we are going to build an interactive tool which allows inputs of body features and desired categories/situations to return
customized recommendations for users. 



## 8 Reference

[1] Rishabh Misra, Mengting Wan, Julian McAuley (2018). Decomposing fit semantics for product size
recommendation in metric spaces.

[2] Hsin-Yu Chen, Huang-Chin Yen, Tian Tian (2020) Female Clothing E-commerce Review and Rating.



Code:

```python
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import json

import gzip
import math
from math import log
import scipy.optimize
import random
import string
import os
import pickle
import itertools
import re
from tqdm import tqdm
import scipy.sparse as sparse
import pyrsm as rsm
import statsmodels.formula.api as smf
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from collections import Counter
from collections import defaultdict

import gensim.models
import gensim.downloader as api
from gensim.test.utils import common_texts
from gensim.models import word2vec
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import logit

import en_core_web_sm
nlp_processor = en_core_web_sm.load()
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem.porter import *

import sklearn
from sklearn import preprocessing
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

from wordcloud import WordCloud
from statsmodels.stats.outliers_influence import variance_inflation_factor
```

    /usr/local/lib/python3.8/dist-packages/gensim/similarities/__init__.py:15: UserWarning: The gensim.similarities.levenshtein submodule is disabled, because the optional Levenshtein package <https://pypi.org/project/python-Levenshtein/> is unavailable. Install Levenhstein (e.g. `pip install python-Levenshtein`) to suppress this warning.
      warnings.warn(msg)



```python
path = "renttherunway_final_data.json"

dataset = []
for line in open(path, 'r'):
    dataset.append(json.loads(line))
    
dataset[0]
```




    {'fit': 'fit',
     'user_id': '420272',
     'bust size': '34d',
     'item_id': '2260466',
     'weight': '137lbs',
     'rating': '10',
     'rented for': 'vacation',
     'review_text': "An adorable romper! Belt and zipper were a little hard to navigate in a full day of wear/bathroom use, but that's to be expected. Wish it had pockets, but other than that-- absolutely perfect! I got a million compliments.",
     'body type': 'hourglass',
     'review_summary': 'So many compliments!',
     'category': 'romper',
     'height': '5\' 8"',
     'size': 14,
     'age': '28',
     'review_date': 'April 20, 2016'}



# Data Pre-process and Explore Data Analysis


```python
df_raw = pd.DataFrame(dataset)
```


```python
# display first 5 rows
df_raw.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }




    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fit</th>
      <th>user_id</th>
      <th>bust size</th>
      <th>item_id</th>
      <th>weight</th>
      <th>rating</th>
      <th>rented for</th>
      <th>review_text</th>
      <th>body type</th>
      <th>review_summary</th>
      <th>category</th>
      <th>height</th>
      <th>size</th>
      <th>age</th>
      <th>review_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>fit</td>
      <td>420272</td>
      <td>34d</td>
      <td>2260466</td>
      <td>137lbs</td>
      <td>10</td>
      <td>vacation</td>
      <td>An adorable romper! Belt and zipper were a lit...</td>
      <td>hourglass</td>
      <td>So many compliments!</td>
      <td>romper</td>
      <td>5' 8"</td>
      <td>14</td>
      <td>28</td>
      <td>April 20, 2016</td>
    </tr>
    <tr>
      <th>1</th>
      <td>fit</td>
      <td>273551</td>
      <td>34b</td>
      <td>153475</td>
      <td>132lbs</td>
      <td>10</td>
      <td>other</td>
      <td>I rented this dress for a photo shoot. The the...</td>
      <td>straight &amp; narrow</td>
      <td>I felt so glamourous!!!</td>
      <td>gown</td>
      <td>5' 6"</td>
      <td>12</td>
      <td>36</td>
      <td>June 18, 2013</td>
    </tr>
    <tr>
      <th>2</th>
      <td>fit</td>
      <td>360448</td>
      <td>NaN</td>
      <td>1063761</td>
      <td>NaN</td>
      <td>10</td>
      <td>party</td>
      <td>This hugged in all the right places! It was a ...</td>
      <td>NaN</td>
      <td>It was a great time to celebrate the (almost) ...</td>
      <td>sheath</td>
      <td>5' 4"</td>
      <td>4</td>
      <td>116</td>
      <td>December 14, 2015</td>
    </tr>
    <tr>
      <th>3</th>
      <td>fit</td>
      <td>909926</td>
      <td>34c</td>
      <td>126335</td>
      <td>135lbs</td>
      <td>8</td>
      <td>formal affair</td>
      <td>I rented this for my company's black tie award...</td>
      <td>pear</td>
      <td>Dress arrived on time and in perfect condition.</td>
      <td>dress</td>
      <td>5' 5"</td>
      <td>8</td>
      <td>34</td>
      <td>February 12, 2014</td>
    </tr>
    <tr>
      <th>4</th>
      <td>fit</td>
      <td>151944</td>
      <td>34b</td>
      <td>616682</td>
      <td>145lbs</td>
      <td>10</td>
      <td>wedding</td>
      <td>I have always been petite in my upper body and...</td>
      <td>athletic</td>
      <td>Was in love with this dress !!!</td>
      <td>gown</td>
      <td>5' 9"</td>
      <td>12</td>
      <td>27</td>
      <td>September 26, 2016</td>
    </tr>
  </tbody>
</table>



</div>




```python
# check columns types
df_raw.dtypes
```




    fit               object
    user_id           object
    bust size         object
    item_id           object
    weight            object
    rating            object
    rented for        object
    review_text       object
    body type         object
    review_summary    object
    category          object
    height            object
    size               int64
    age               object
    review_date       object
    dtype: object




```python
# check the proportion of null values
df_raw.isnull().sum()/df_raw.shape[0]
```




    fit               0.000000
    user_id           0.000000
    bust size         0.095620
    item_id           0.000000
    weight            0.155715
    rating            0.000426
    rented for        0.000052
    review_text       0.000000
    body type         0.076019
    review_summary    0.000000
    category          0.000000
    height            0.003516
    size              0.000000
    age               0.004986
    review_date       0.000000
    dtype: float64




```python
# get modes of columns to fill in missing values
col_feature = ["bust size","weight","body type","height","size","age"]
```


```python
df_raw.loc[:,col_feature].mode()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }




    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>bust size</th>
      <th>weight</th>
      <th>body type</th>
      <th>height</th>
      <th>size</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34b</td>
      <td>130lbs</td>
      <td>hourglass</td>
      <td>5' 4"</td>
      <td>8</td>
      <td>31</td>
    </tr>
  </tbody>
</table>



</div>




```python
# data pre-processing
for i in dataset:
    if 'rented for' not in i.keys():
        dataset.remove(i)
    elif (i['rating'] == None) or (len(i['review_text']) == 0) or (len(i['fit']) == 0) or (len(i['category']) == 0) or (len(i['rented for']) == 0): 
        dataset.remove(i)
len(dataset)
```




    192452




```python
bust_size =  []
for d in dataset:
    if 'bust size' in d.keys():
        bust_size.append(d['bust size'])
    else:
        bust_size.append(0)
bust_number = []
cup = []
for b in bust_size:
    if b != 0 :
        bust_number.append(int(b[0:2]))
        cup.append(b[2:])
    else:
        bust_number.append(34)
        cup.append('b')
body_type =  []
for d in dataset:
    if 'body type' in d.keys():
        body_type.append(d['body type'])
    else:
        body_type.append('hourglass')
weight =  []
for d in dataset:
    if 'weight' in d.keys():
        weight.append(int(d['weight'].split('l',1)[0]))
    else:
        weight.append(130)
height =  []
for d in dataset:
    if 'height' in d.keys():
        height.append(12*int(d['height'].split("\' ")[0])+int(d['height'].split("\' ")[1].split('"')[0]))
    else:
        height.append(64)
size =  []
for d in dataset:
    if 'size' in d.keys():
        size.append(d['size'])
    else:
        size.append(8)
age =  []
for d in dataset:
    if 'age' in d.keys():
        age.append(int(d['age']))
    else:
        age.append(31)
fit = [d['fit'] for d in dataset]
rate = [(d['rating']) for d in dataset]
category = [d['category'] for d in dataset]
rent =  []
for d in dataset:
    if 'rented for' in d.keys():
        rent.append(d['rented for'])
    else:
        rent.append('Unknown')
reviews = [d['review_text'] for d in dataset]
```


```python
df_clean = pd.DataFrame({'fit': fit, 'rating': rate, 'category': category, 'rent_for': rent, 
                  'bust_number': bust_number, 'cup': cup, 'body_type': body_type, 'weight' : weight, 'height': height, 
                  'size': size, 'age': age, 'review_text': reviews})
```


```python
df_clean.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }




    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fit</th>
      <th>rating</th>
      <th>category</th>
      <th>rent_for</th>
      <th>bust_number</th>
      <th>cup</th>
      <th>body_type</th>
      <th>weight</th>
      <th>height</th>
      <th>size</th>
      <th>age</th>
      <th>review_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>fit</td>
      <td>10</td>
      <td>romper</td>
      <td>vacation</td>
      <td>34</td>
      <td>d</td>
      <td>hourglass</td>
      <td>137</td>
      <td>68</td>
      <td>14</td>
      <td>28</td>
      <td>An adorable romper! Belt and zipper were a lit...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>fit</td>
      <td>10</td>
      <td>gown</td>
      <td>other</td>
      <td>34</td>
      <td>b</td>
      <td>straight &amp; narrow</td>
      <td>132</td>
      <td>66</td>
      <td>12</td>
      <td>36</td>
      <td>I rented this dress for a photo shoot. The the...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>fit</td>
      <td>10</td>
      <td>sheath</td>
      <td>party</td>
      <td>34</td>
      <td>b</td>
      <td>hourglass</td>
      <td>130</td>
      <td>64</td>
      <td>4</td>
      <td>116</td>
      <td>This hugged in all the right places! It was a ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>fit</td>
      <td>8</td>
      <td>dress</td>
      <td>formal affair</td>
      <td>34</td>
      <td>c</td>
      <td>pear</td>
      <td>135</td>
      <td>65</td>
      <td>8</td>
      <td>34</td>
      <td>I rented this for my company's black tie award...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>fit</td>
      <td>10</td>
      <td>gown</td>
      <td>wedding</td>
      <td>34</td>
      <td>b</td>
      <td>athletic</td>
      <td>145</td>
      <td>69</td>
      <td>12</td>
      <td>27</td>
      <td>I have always been petite in my upper body and...</td>
    </tr>
  </tbody>
</table>



</div>




```python
# Plotting barplots for response variables
# increase plot resolution
mpl.rcParams["figure.dpi"] = 80
```


```python
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)

df_clean.fit.value_counts(normalize=True,ascending=True).plot(kind='barh',ax=ax1)
df_clean.rating.value_counts(normalize=True,ascending=True).plot(kind='barh',ax=ax2)
df_clean.rent_for.value_counts(normalize=True,ascending=True).plot(kind='barh',ax=ax3)
category_sort = df_clean.category.value_counts().reset_index().sort_values(by=['category'],ascending=False)
category_top10 = list(category_sort['index'][:10])
df_clean[df_clean.category.isin(category_top10)].category.value_counts(normalize=True,ascending=True).plot(kind='barh',ax=ax4)
#fig.suptitle('Proportion of Variables',y=1.0)
ax1.set_title('Fit')
ax2.set_title('Rating')
ax3.set_title('Rented for')
ax4.set_title('Category')
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
plt.show()
```


![png](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/output_14_0.png)



```python
fig, (ax1, ax2) = plt.subplots(2)
df_clean.cup.value_counts(normalize=True,ascending=False).plot(kind='bar',ax=ax1, rot=0)
df_clean.body_type.value_counts(normalize=True,ascending=False).plot(kind='bar',ax=ax2, rot=30)
#fig.suptitle('Proportion of Categorical Features',y=1.0)
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
ax1.set_title('Cup')
ax2.set_title('Body Type')
plt.show()
```


![png](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/output_15_0.png)



```python
#mpl.rcParams["figure.dpi"] = 80
fig, axes = plt.subplots(1,3)
sns.boxplot(y=df_clean.bust_number,ax=axes[0])
sns.boxplot(y=df_clean.weight,ax=axes[1])
sns.boxplot(y=df_clean.height,ax=axes[2])
#fig.suptitle('Boxplot of Features',y=1.0)
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.6, 
                    hspace=0.4)
plt.show()
```


![png](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/output_16_0.png)



```python
#mpl.rcParams["figure.dpi"] = 80
fig, axes = plt.subplots(1,2)
sns.boxplot(y=df_clean["age"],ax=axes[0])
sns.boxplot(y=df_clean["size"],ax=axes[1])
#fig.suptitle('Boxplot of Features',y=1.0)
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.6, 
                    hspace=0.4)
plt.show()
```


![png](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/output_17_0.png)


#### Since there are unreasonable value in age, we decided to change those <15 and >70 to mode(31)


```python
df_clean.loc[(df_clean['age'] > 70) | (df_clean['age'] < 15), 'age'] = 31
```


```python
mpl.rcParams["figure.dpi"] = 80
sns.pairplot(data = df_clean, vars=["bust_number","weight","height","age"])
plt.show()
```


![png](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/output_20_0.png)



```python
#mpl.rcParams["figure.dpi"] = 150
# Creating a matrix using age, salry, balance as rows and columns
df_clean[['weight','height','bust_number']].corr()

#plot the correlation matrix of salary, balance and age in data dataframe.
sns.heatmap(df_clean[['weight','height','bust_number']].corr(), annot=True, cmap = 'Reds')
plt.show()
```


![png](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/output_21_0.png)



```python
# Output wordcloud
wordCount = defaultdict(int)
punctuation = set(string.punctuation)
stemmer = PorterStemmer()
stopWords = stopwords.words('english')
```


```python
review_summary = []
for d in dataset:
    r = ''.join([c for c in d['review_summary'].lower()])
    for w in r.split():
        #w = stemmer.stem(w)
        if (w not in punctuation) and (w not in stopWords):
            review_summary.append(w)
```


```python
nltk.download('stopwords')
stop = set(stopwords.words('english'))

ps = PorterStemmer() 

# return a list of tokens
def pre_processing_by_nltk(doc, stemming = True, need_sent = False):
    # step 1: get sentences
    sentences = sent_tokenize(doc)
    # step 2: get tokens
    tokens = []
    for sent in sentences:
        words = word_tokenize(sent)
        # step 3: stemming
        if stemming:
            words = [ps.stem(word) for word in words]
        if need_sent:
            tokens.append(words)
        else:
            tokens += words
    # remove stop words and punctuations in reutrns
    return [w.lower() for w in tokens if w.lower() not in stop if w.isalnum()]
```

    [nltk_data] Downloading package stopwords to /home/jovyan/nltk_data...
    [nltk_data]   Package stopwords is already up-to-date!



```python
review_text = np.array(df_clean["review_text"].agg(pre_processing_by_nltk, stemming = False, need_sent = False))
review_text = review_text.reshape(-1).tolist()
review_text_lst = [i for lst in review_text for i in lst]
reviewtext_dict=Counter(review_text_lst)
reviewtext_wordcloud = WordCloud(width = 1000, height = 500, background_color='white').generate_from_frequencies(reviewtext_dict)

plt.figure(figsize=(15,8))
plt.imshow(reviewtext_wordcloud)
plt.axis("off")
plt.show()
```


![png](https://gitee.com/flycloud2009_cloudlou/img/raw/master/img/output_25_0.png)


# Use 50K data choose model


```python
df_clean.shape
```




    (192452, 12)




```python
dataset_50k = df_clean.sample(frac=1,random_state=42).reset_index(drop=True)[:50000]
dataset_50k.shape
```




    (50000, 12)



# Baseline (without review_text)


```python
dataset_50k.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }




    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>fit</th>
      <th>rating</th>
      <th>category</th>
      <th>rent_for</th>
      <th>bust_number</th>
      <th>cup</th>
      <th>body_type</th>
      <th>weight</th>
      <th>height</th>
      <th>size</th>
      <th>age</th>
      <th>review_text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>large</td>
      <td>10</td>
      <td>maxi</td>
      <td>party</td>
      <td>34</td>
      <td>b</td>
      <td>pear</td>
      <td>140</td>
      <td>64</td>
      <td>8</td>
      <td>36</td>
      <td>I straddle the 4/6 sizes and this fit me perfe...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>fit</td>
      <td>10</td>
      <td>gown</td>
      <td>formal affair</td>
      <td>32</td>
      <td>d</td>
      <td>hourglass</td>
      <td>130</td>
      <td>63</td>
      <td>8</td>
      <td>31</td>
      <td>I didn't want to send it back!  The 4 fit perf...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>fit</td>
      <td>10</td>
      <td>sheath</td>
      <td>formal affair</td>
      <td>32</td>
      <td>d</td>
      <td>athletic</td>
      <td>142</td>
      <td>68</td>
      <td>12</td>
      <td>30</td>
      <td>Fit great, if you are between sizes order up, ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>fit</td>
      <td>10</td>
      <td>dress</td>
      <td>wedding</td>
      <td>38</td>
      <td>c</td>
      <td>apple</td>
      <td>175</td>
      <td>67</td>
      <td>20</td>
      <td>31</td>
      <td>I went out on a limb with this dress - all the...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>fit</td>
      <td>8</td>
      <td>midi</td>
      <td>wedding</td>
      <td>36</td>
      <td>ddd/e</td>
      <td>full bust</td>
      <td>170</td>
      <td>69</td>
      <td>24</td>
      <td>31</td>
      <td>It is low, although you're covered you can see...</td>
    </tr>
  </tbody>
</table>



</div>




```python
df = pd.get_dummies(dataset_50k, columns = ['cup','body_type'])
```


```python
X = df[['bust_number', 'weight',
       'height', 'size', 'age', 'cup_a', 'cup_aa', 'cup_b',
       'cup_c', 'cup_d', 'cup_d+', 'cup_dd', 'cup_ddd/e', 'cup_f', 'cup_g',
       'cup_h', 'cup_i', 'cup_j', 'body_type_apple', 'body_type_athletic',
       'body_type_full bust', 'body_type_hourglass', 'body_type_pear',
       'body_type_petite', 'body_type_straight & narrow']]
X_fit_train, X_fit_test, y_fit_train, y_fit_test = train_test_split(X, df.loc[:,'fit'], random_state=42, test_size=0.2, shuffle=True)
X_rate_train, X_rate_test, y_rate_train, y_rate_test = train_test_split(X,df.loc[:,'rating'], random_state=42, test_size=0.2, shuffle=True)
```


```python
# Test for multicollinearity
X['intercept'] =1
vif = variance_inflation_factor(X.values, 1)
vif
```

    <ipython-input-21-c2510b5486bd>:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      X['intercept'] =1





    2.2781909944415895




```python
def eval(model, x, true):
    auroc = roc_auc_score(true, model.predict_proba(x), multi_class = 'ovr')
    f1_mac = f1_score(true, model.predict(x), average='macro')
    f1_mic = f1_score(true, model.predict(x), average='micro')
    accuracy = accuracy_score(true, model.predict(x))
    print("auroc_rate: ", auroc)
    print("f1_mac_rate: ", f1_mac)
    print("f1_mic_rate: ", f1_mic)
    print("accuracy_rate: ", accuracy)
    
    return(auroc, f1_mac, f1_mic, accuracy)
```


```python
X_fit_train_sparse = sparse.csr_matrix(X_fit_train)
X_rate_train_sparse = sparse.csr_matrix(X_rate_train)
```


```python
clf_fit_base = LogisticRegression(max_iter=100000000,C=1).fit(X_fit_train, y_fit_train)
auroc_fit_base, f1_mac_fit_base, f1_mic_fit_base, accuracy_fit_base = eval(clf_fit_base, sparse.csr_matrix(X_fit_test), y_fit_test)
```

    auroc_rate:  0.5866287174095914
    f1_mac_rate:  0.28463118112481195
    f1_mic_rate:  0.7338999999999999
    accuracy_rate:  0.7339



```python
clf_rate_base = LogisticRegression(max_iter=100000000,C=1).fit(X_rate_train, y_rate_train)
auroc_rate_base, f1_mac_rate_base, f1_mic_rate_base, accuracy_rate_base = eval(clf_rate_base,sparse.csr_matrix( X_rate_test), y_rate_test)
```

    auroc_rate:  0.5455740291115136
    f1_mac_rate:  0.15766387980128438
    f1_mic_rate:  0.6506
    accuracy_rate:  0.6506


# Add Text

## TFIDF


```python
# Text Pre-Processing
stop_words = set(stopwords.words('english'))
newStopWords = ['actually', 'also', 'although',
    'always', 'became', 'become', 'could','do', 'does', 'either', 'else','hence','how', 'may', 'to', 'on'
    'maybe','me','might','mine','must','my','of','oh','ok','whereas','wherever',
    'whenever','whether','whoever','whose','why','will','within','with','without','yet','your', 'would', 'all']
stop_words.update(newStopWords)

# print(stop) must be lowercase an √ ； AN X
ps = PorterStemmer() 

# return a list of tokens
def pre_processing_by_nltk(doc, stemming = True, need_sent = False):
    # step 1: get sentences
    sentences = sent_tokenize(doc)
    # step 2: get tokens
    tokens = []
    for sent in sentences:
        words = word_tokenize(sent)
        # step 3: stemming
        if stemming:
            words = [ps.stem(word) for word in words]
        if need_sent:
            tokens.append(words)
        else:
            tokens += words
    return [w.lower() for w in tokens if w.lower() not in stop_words]

# Create vocab
DF = defaultdict(float)
for doc in tqdm(df.review_text):
    tokens = pre_processing_by_nltk(doc)
    for token in set(tokens):
        DF[token] += 1
        
IDF, vocab = dict(), dict()
for token in DF:
    if DF[token] < 15:
        # words (freq < 10) becomes an unk
        pass
    else:
        vocab[token] = len(vocab)
        IDF[token] = log(1 + len(df.review_text) / DF[token])

IDF['<UNK>'] = 1
vocab['<UNK>'] = len(vocab)

len(vocab)
```

    100%|██████████| 50000/50000 [01:26<00:00, 579.70it/s]





    2951




```python
def tfidf_feature_extractor(doc, vocab, IDF):
    tokens = pre_processing_by_nltk(doc)
    for i, token in enumerate(tokens):
        if token not in vocab:
            tokens[i] = '<UNK>'
    TF = defaultdict(int)
    for token in tokens:
        TF[token] += 1
    x = [0] * len(vocab)
    for token in set(tokens):
        tfidf = log(TF[token] + 1) * IDF[token]
        token_id = vocab[token]
#         print(token, TF[token], IDF[token])
        x[token_id] = tfidf # this will be a dense matrix
    return x
```


```python
X_review_tfidf = []
for doc in tqdm(df.review_text):
    X_review_tfidf.append(tfidf_feature_extractor(doc, vocab, IDF))
X_review = pd.DataFrame(np.array(X_review_tfidf))
```

    100%|██████████| 50000/50000 [01:33<00:00, 532.47it/s]



```python
X_add_idf = pd.concat([X,X_review],axis=1)
```


```python
X_fit_train, X_fit_test, y_fit_train, y_fit_test = train_test_split(X_add_idf, df.loc[:,'fit'], random_state=42, test_size=0.2, shuffle=True)
X_rate_train, X_rate_test, y_rate_train, y_rate_test = train_test_split(X_add_idf,df.loc[:,'rating'], random_state=42, test_size=0.2, shuffle=True)
X_fit_train_sparse = sparse.csr_matrix(X_fit_train)
X_rate_train_sparse = sparse.csr_matrix(X_rate_train)
```


```python
clf_fit_idf = LogisticRegression(max_iter=100000000,C=1).fit(X_fit_train_sparse, y_fit_train)
auroc_fit_idf, f1_mac_fit_idf, f1_mic_fit_idf, accuracy_fit_idf = eval(clf_fit_idf, sparse.csr_matrix(X_fit_test), y_fit_test)
```

    auroc_rate:  0.790526609454432
    f1_mac_rate:  0.6030773228208012
    f1_mic_rate:  0.7731
    accuracy_rate:  0.7731



```python
clf_rate_idf = LogisticRegression(max_iter=100000000,C=1).fit(X_rate_train_sparse, y_rate_train)
auroc_rate_idf, f1_mac_rate_idf, f1_mic_rate_idf, accuracy_rate_idf = eval(clf_rate_idf, sparse.csr_matrix(X_rate_test), y_rate_test)
```

    auroc_rate:  0.7883374631115887
    f1_mac_rate:  0.34772206797681615
    f1_mic_rate:  0.6721
    accuracy_rate:  0.6721


## Word2Vec


```python
def build_vocab(sentences):
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return word_counts, vocabulary, vocabulary_inv

def get_embeddings(inp_data, vocabulary_inv, size_features=100,
                   mode='skipgram',
                   min_word_count=2,
                   context=5):
    model_name = "embedding"
    model_name = os.path.join(model_name)
    num_workers = 15  # Number of threads to run in parallel
    downsampling = 1e-3  # Downsample setting for frequent words
    print('Training Word2Vec model...')
    # use inp_data and vocabulary_inv to reconstruct sentences
    sentences = [[vocabulary_inv[w] for w in s] for s in inp_data]
    if mode == 'skipgram':
        sg = 1
        print('Model: skip-gram')
    elif mode == 'cbow':
        sg = 0
        print('Model: CBOW')
    embedding_model = word2vec.Word2Vec(sentences, workers=num_workers,
                                        sg=sg,
                                        vector_size=size_features,
                                        min_count=min_word_count,
                                        window=context,
                                        sample=downsampling)
    print("Saving Word2Vec model {}".format(model_name))
    embedding_weights = np.zeros((len(vocabulary_inv), size_features))
    for i in range(len(vocabulary_inv)):
        word = vocabulary_inv[i]
        if word in embedding_model.wv:
            embedding_weights[i] = embedding_model.wv[word]
        else:
            embedding_weights[i] = np.random.uniform(-0.25, 0.25,
                                                     embedding_model.vector_size)
    return embedding_weights

def preprocess_df(df):
    # get English stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.add('would')
    # prepare translation table to translate punctuation to space
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    preprocessed_sentences = []
    for i, row in df.iterrows():
        sent = row["review_text"]
        sent_nopuncts = sent.translate(translator)
        words_list = sent_nopuncts.strip().split()
        filtered_words = [word for word in words_list if word not in stop_words and len(word) != 1] # also skip space from above translation
        preprocessed_sentences.append(" ".join(filtered_words))
    df["review_text"] = preprocessed_sentences
    return df
```


```python
train = df[:len(df)//10*8]
test = df[len(df)//10*8:]

train_feature = X[:len(X)//10*8]
test_feature = X[len(X)//10*8:]

train = preprocess_df(train)
test = preprocess_df(test)
```

    <ipython-input-41-424ecf821ad2>:57: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df["review_text"] = preprocessed_sentences



```python
# tokenization 
tagged_data = [word_tokenize(_d) for i, _d in enumerate(train["review_text"])]
# build vocabulary from tokenized data
word_counts, vocabulary, vocabulary_inv = build_vocab(tagged_data)
# use the above mapping to create input data
inp_data = [[vocabulary[word] for word in text] for text in tagged_data]
# get embedding vector
embedding_weights = get_embeddings(inp_data, vocabulary_inv)

tagged_train_data = [word_tokenize(_d) for i, _d in enumerate(train["review_text"])]
tagged_test_data = [word_tokenize(_d) for i, _d in enumerate(test["review_text"])]
```

    Training Word2Vec model...
    Model: skip-gram
    Saving Word2Vec model embedding



```python
train_vec = []
for doc in tagged_train_data:
    vec = 0
    for w in doc:
        vec += embedding_weights[vocabulary[w]]
    if len(doc) != 0:
        vec = vec / len(doc)
    else: vec = [0] * 100
    train_vec.append(vec)

test_vec = []
for doc in tagged_test_data:
    vec = 0
    length = 0
    for w in doc:
        try:
            vec += embedding_weights[vocabulary[w]]
            length += 1
        except:
            continue
    if length != 0: 
        vec = vec / length
    else: vec = [0] * 100
    test_vec.append(vec)
```


```python
train_vec_features = [np.append(i, j) for (i,j) in zip(train_vec, np.array(train_feature))]
test_vec_features = [np.append(i, j) for (i,j) in zip(test_vec, np.array(test_feature))]

train_just_features = [i for i in np.array(train_feature)]
test_just_features = [i for i in np.array(test_feature)]
```


```python
model_fit_SG = LogisticRegression(C = 1, random_state=42, max_iter = 100000000).fit(train_vec_features, train['fit'])
auroc_fit_SG, f1_mac_fit_SG, f1_mic_fit_SG, accuracy_fit_SG  = eval(model_fit_SG, test_vec_features, test['fit'])
```

    auroc_rate:  0.7852504496310196
    f1_mac_rate:  0.48416385275120405
    f1_mic_rate:  0.7669
    accuracy_rate:  0.7669



```python
model_rating_SG = LogisticRegression(C = 1, random_state=42, max_iter = 100000000).fit(train_vec_features, train['rating'])
auroc_rate_SG, f1_mac_rate_SG, f1_mic_rate_SG, accuracy_rate_SG = eval(model_rating_SG, test_vec_features, test['rating'])
```

    auroc_rate:  0.8226084393207364
    f1_mac_rate:  0.26879388027214113
    f1_mic_rate:  0.6763
    accuracy_rate:  0.6763


## CBOW


```python
# get embedding vector
embedding_weights = get_embeddings(inp_data, vocabulary_inv, mode='cbow' )

tagged_train_data = [word_tokenize(_d) for i, _d in enumerate(train["review_text"])]
tagged_test_data = [word_tokenize(_d) for i, _d in enumerate(test["review_text"])]
```

    Training Word2Vec model...
    Model: CBOW
    Saving Word2Vec model embedding



```python
train_vec = []
for doc in tagged_train_data:
    vec = 0
    for w in doc:
        vec += embedding_weights[vocabulary[w]]
    if len(doc) != 0:
        vec = vec / len(doc)
    else: vec = [0] * 100
    train_vec.append(vec)

test_vec = []
for doc in tagged_test_data:
    vec = 0
    length = 0
    for w in doc:
        try:
            vec += embedding_weights[vocabulary[w]]
            length += 1
        except:
            continue
    if length != 0: 
        vec = vec / length
    else: vec = [0] * 100
    test_vec.append(vec)
```


```python
train_vec_features = [np.append(i, j) for (i,j) in zip(train_vec, np.array(train_feature))]
test_vec_features = [np.append(i, j) for (i,j) in zip(test_vec, np.array(test_feature))]
```


```python
model_fit_cbow = LogisticRegression(C = 1, random_state=42, max_iter = 100000000).fit(train_vec_features, train['fit'])
auroc_fit_cbow, f1_mac_fit_cbow, f1_mic_fit_cbow, accuracy_fit_cbow  = eval(model_fit_cbow, test_vec_features, test['fit'])
```

    auroc_rate:  0.7948469125828606
    f1_mac_rate:  0.508236085791458
    f1_mic_rate:  0.7708
    accuracy_rate:  0.7708



```python
model_rating_cbow = LogisticRegression(C = 1, random_state=42, max_iter = 100000000).fit(train_vec_features, train['rating'])
auroc_rate_cbow, f1_mac_rate_cbow, f1_mic_rate_cbow, accuracy_rate_cbow = eval(model_rating_cbow, test_vec_features, test['rating'])
```

    auroc_rate:  0.810268130641866
    f1_mac_rate:  0.25538126948925366
    f1_mic_rate:  0.6646
    accuracy_rate:  0.6646


## Glove


```python
tagged_train_data_list = []
for i in tagged_train_data:
    tagged_train_data_list.append(' '.join(i))
    
tagged_test_data_list = []
for i in tagged_test_data:
    tagged_test_data_list.append(' '.join(i))
glove_df_train = pd.DataFrame(tagged_train_data_list).rename(columns={0:'text'})
glove_df_test = pd.DataFrame(tagged_test_data_list).rename(columns={0:'text'})
```


```python
glove_model = {}
with open ('glove.6B.100d.txt') as f:
    for line in f:
        line_split = line.split()
        word = line_split[0]
        embedding = np.array(line_split[1:], dtype=np.float64)  
        glove_model[word] = embedding
```


```python
glove_model_new = {}
for k, v in glove_model.items():
    if v.size ==100:
        glove_model_new[k] = v
```


```python
def cos_sim(a, b):
    return dot(a, b)/(norm(a)*norm(b))

def GloVe(doc, wv):
    vecs = []
    for token in doc.split():
        try:
            vecs.append(wv[token])
        except KeyError:
            vecs.append(np.array([0]*100))
    return np.mean(vecs, axis=0)
```


```python
train_vec = []
for sent in glove_df_train.text:
    train_vec.append(GloVe(sent, glove_model_new))

test_vec = []
for sent in glove_df_test.text:
    test_vec.append(GloVe(sent, glove_model_new))
```

    /usr/local/lib/python3.8/dist-packages/numpy/core/fromnumeric.py:3419: RuntimeWarning: Mean of empty slice.
      return _methods._mean(a, axis=axis, dtype=dtype,
    /usr/local/lib/python3.8/dist-packages/numpy/core/_methods.py:188: RuntimeWarning: invalid value encountered in double_scalars
      ret = ret.dtype.type(ret / rcount)



```python
train_vec_features_df = pd.DataFrame(train_vec_features)
train_vec_features_df = pd.concat([train_vec_features_df, train[['fit', 'rating']]], axis=1)
test_vec_features_df = pd.DataFrame(test_vec_features)
test_vec_features_df = pd.concat([test_vec_features_df, test[['fit', 'rating']].reset_index()], axis=1)
train_vec_features_df = train_vec_features_df.dropna()
test_vec_features_df = test_vec_features_df.dropna()
test_vec_features_df = test_vec_features_df.drop(columns=['index'])
```


```python
model_fit_Glove = LogisticRegression(C = 1, random_state=42, max_iter = 100000000).fit(train_vec_features_df.iloc[:,:-2], train_vec_features_df['fit'])
auroc_fit_Glove, f1_mac_fit_Glove, f1_mic_fit_Glove, accuracy_fit_Glove  = eval(model_fit_Glove, test_vec_features_df.iloc[:,:-2], test_vec_features_df['fit'])
```

    auroc_rate:  0.7500948658707619
    f1_mac_rate:  0.4330966750550509
    f1_mic_rate:  0.7553308639503454
    accuracy_rate:  0.7553308639503454



```python
model_rate_Glove = LogisticRegression(C = 1, random_state=42, max_iter = 100000000).fit(train_vec_features_df.iloc[:,:-2], train_vec_features_df['rating'])
auroc_rate_Glove, f1_mac_rate_Glove, f1_mic_rate_Glove, accuracy_rate_Glove  = eval(model_rate_Glove, test_vec_features_df.iloc[:,:-2], test_vec_features_df['rating'])
```

    auroc_rate:  0.754920472670829
    f1_mac_rate:  0.19990658444584186
    f1_mic_rate:  0.649614576033637
    accuracy_rate:  0.649614576033637


# Compare Model


```python
# fit
compare_fit = pd.DataFrame({
    'baseline':[auroc_fit_base, f1_mac_fit_base, f1_mic_fit_base, accuracy_fit_base],
    'TFIDF':[auroc_fit_idf, f1_mac_fit_idf, f1_mic_fit_idf, accuracy_fit_idf],
    'skip-gram':[auroc_fit_SG, f1_mac_fit_SG, f1_mic_fit_SG, accuracy_fit_SG],
    'CBOW':[auroc_fit_cbow, f1_mac_fit_cbow, f1_mic_fit_cbow, accuracy_fit_cbow],
    'Glove':[auroc_fit_Glove, f1_mac_fit_Glove, f1_mic_fit_Glove, accuracy_fit_Glove]
},index = ['auroc','f1_mac','f1_mic','accuracy'])
compare_fit.round(4)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }




    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>baseline</th>
      <th>TFIDF</th>
      <th>skip-gram</th>
      <th>CBOW</th>
      <th>Glove</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>auroc</th>
      <td>0.5866</td>
      <td>0.7905</td>
      <td>0.7853</td>
      <td>0.7948</td>
      <td>0.7501</td>
    </tr>
    <tr>
      <th>f1_mac</th>
      <td>0.2846</td>
      <td>0.6031</td>
      <td>0.4842</td>
      <td>0.5082</td>
      <td>0.4331</td>
    </tr>
    <tr>
      <th>f1_mic</th>
      <td>0.7339</td>
      <td>0.7731</td>
      <td>0.7669</td>
      <td>0.7708</td>
      <td>0.7553</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.7339</td>
      <td>0.7731</td>
      <td>0.7669</td>
      <td>0.7708</td>
      <td>0.7553</td>
    </tr>
  </tbody>
</table>


</div>




```python
# rate
compare_rate = pd.DataFrame({
    'baseline':[auroc_rate_base, f1_mac_rate_base, f1_mic_rate_base, accuracy_rate_base],
    'TFIDF':[auroc_rate_idf, f1_mac_rate_idf, f1_mic_rate_idf, accuracy_rate_idf],
    'skip-gram':[auroc_rate_SG, f1_mac_rate_SG, f1_mic_rate_SG, accuracy_rate_SG],
    'CBOW':[auroc_rate_cbow, f1_mac_rate_cbow, f1_mic_rate_cbow, accuracy_rate_cbow],
    'Glove':[auroc_rate_Glove, f1_mac_rate_Glove, f1_mic_rate_Glove, accuracy_rate_Glove]
},index = ['auroc','f1_mac','f1_mic','accuracy'])
compare_rate.round(4)
```






<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }




    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>baseline</th>
      <th>TFIDF</th>
      <th>skip-gram</th>
      <th>CBOW</th>
      <th>Glove</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>auroc</th>
      <td>0.5456</td>
      <td>0.7883</td>
      <td>0.8226</td>
      <td>0.8103</td>
      <td>0.7549</td>
    </tr>
    <tr>
      <th>f1_mac</th>
      <td>0.1577</td>
      <td>0.3477</td>
      <td>0.2688</td>
      <td>0.2554</td>
      <td>0.1999</td>
    </tr>
    <tr>
      <th>f1_mic</th>
      <td>0.6506</td>
      <td>0.6721</td>
      <td>0.6763</td>
      <td>0.6646</td>
      <td>0.6496</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.6506</td>
      <td>0.6721</td>
      <td>0.6763</td>
      <td>0.6646</td>
      <td>0.6496</td>
    </tr>
  </tbody>
</table>


</div>



### Then we choose CBOW as our final method to process text data


```python
df_full = df_clean.copy()
```


```python
df_full = pd.get_dummies(df_full, columns = ['cup','body_type'])
df_feature_full = df_full[['bust_number', 'weight',
       'height', 'size', 'age', 'cup_a', 'cup_aa', 'cup_b',
       'cup_c', 'cup_d', 'cup_d+', 'cup_dd', 'cup_ddd/e', 'cup_f', 'cup_g',
       'cup_h', 'cup_i', 'cup_j', 'body_type_apple', 'body_type_athletic',
       'body_type_full bust', 'body_type_hourglass', 'body_type_pear',
       'body_type_petite', 'body_type_straight & narrow']]
```


```python
train_full = df_full[:len(df_full)//10*8]
test_full = df_full[len(df_full)//10*8:]

train_feature_full = df_feature_full[:len(df_feature_full)//10*8]
test_feature_full = df_feature_full[len(df_feature_full)//10*8:]

train_full = preprocess_df(train_full)
test_full = preprocess_df(test_full)
```

    <ipython-input-41-424ecf821ad2>:57: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df["review_text"] = preprocessed_sentences



```python
# tokenization 
tagged_data_full = [word_tokenize(_d) for i, _d in enumerate(train_full["review_text"])]
# build vocabulary from tokenized data
word_counts_full, vocabulary_full, vocabulary_inv_full = build_vocab(tagged_data_full)
# use the above mapping to create input data
inp_data_full = [[vocabulary_full[word] for word in text] for text in tagged_data_full]
# get embedding vector
embedding_weights_full = get_embeddings(inp_data_full, vocabulary_inv_full,mode='cbow')

tagged_train_data_full = [word_tokenize(_d) for i, _d in enumerate(train_full["review_text"])]
tagged_test_data_full = [word_tokenize(_d) for i, _d in enumerate(test_full["review_text"])]
```

    Training Word2Vec model...
    Model: CBOW
    Saving Word2Vec model embedding



```python
train_vec_full = []
for doc in tagged_train_data_full:
    vec = 0
    for w in doc:
        vec += embedding_weights_full[vocabulary_full[w]]
    if len(doc) != 0:
        vec = vec / len(doc)
    else: vec = [0] * 100
    train_vec_full.append(vec)

test_vec_full = []
for doc in tagged_test_data_full:
    vec = 0
    length = 0
    for w in doc:
        try:
            vec += embedding_weights_full[vocabulary_full[w]]
            length += 1
        except:
            continue
    if length != 0: 
        vec = vec / length
    else: vec = [0] * 100
    test_vec_full.append(vec)
```


```python
train_vec_features_full = [np.append(i, j) for (i,j) in zip(train_vec_full, np.array(train_feature_full))]
test_vec_features_full = [np.append(i, j) for (i,j) in zip(test_vec_full, np.array(test_feature_full))]

train_just_features_full = [i for i in np.array(train_feature_full)]
test_just_features_full = [i for i in np.array(test_feature_full)]
```


```python
model_rating_baseline_full = LogisticRegression(C =1, random_state=42, max_iter = 100000000).fit(train_just_features_full, train_full['rating'])
auroc_rating_baseline_full, f1_mac_rating_baseline_full, f1_mic_rating_baseline_full, accuracy_rating_baseline_full = eval(model_rating_baseline_full, test_just_features_full, test_full['rating'])
```

    auroc_rate:  0.5339586064760585
    f1_mac_rate:  0.1569258943512993
    f1_mic_rate:  0.6455886937545464
    accuracy_rate:  0.6455886937545464



```python
model_rating_full = LogisticRegression(C = 1, random_state=42, max_iter = 100000000).fit(train_vec_features_full, train_full['rating'])
auroc_rating_CBOW_full, f1_mac_rating_CBOW_full, f1_mic_rating_CBOW_full, accuracy_rating_CBOW_full=eval(model_rating_full, test_vec_features_full, test_full['rating'])
```

    auroc_rate:  0.8207798206398245
    f1_mac_rate:  0.27803024060448356
    f1_mic_rate:  0.6739582250857321
    accuracy_rate:  0.6739582250857321



```python
model_fit_baseline_full = LogisticRegression(C = 1, random_state=42, max_iter = 100000000).fit(train_just_features_full, train_full['fit'])
auroc_fit_baseline_full, f1_mac_fit_baseline_full, f1_mic_fit_baseline_full, accuracy_fit__baseline_full= eval(model_fit_baseline_full, test_just_features_full, test_full['fit'])
```

    auroc_rate:  0.5856004766883958
    f1_mac_rate:  0.285921879398892
    f1_mic_rate:  0.7369583290034293
    accuracy_rate:  0.7369583290034293



```python
model_fit_full = LogisticRegression(C = 1, random_state=42, max_iter = 100000000).fit(train_vec_features_full, train_full['fit'])
auroc_fit_CBOW_full, f1_mac_fit_CBOW_full, f1_mic_fit_CBOW_full, accuracy_fit_CBOW_full=eval(model_fit_full, test_vec_features_full, test_full['fit'])
```

    auroc_rate:  0.7880492150879822
    f1_mac_rate:  0.5023983522203012
    f1_mic_rate:  0.7683934324015379
    accuracy_rate:  0.768393432401538



```python
# fit
compare_full_fit = pd.DataFrame({
    'baseline_full':[auroc_fit_baseline_full, f1_mac_fit_baseline_full, f1_mic_fit_baseline_full, accuracy_fit__baseline_full],
    'CBOW_full':[auroc_fit_CBOW_full, f1_mac_fit_CBOW_full, f1_mic_fit_CBOW_full, accuracy_fit_CBOW_full]
},index = ['auroc','f1_mac','f1_mic','accuracy'])
compare_full_fit['improvement'] =  (compare_full_fit['CBOW_full']-compare_full_fit['baseline_full'])
compare_full_fit.round(4)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }




    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>baseline_full</th>
      <th>CBOW_full</th>
      <th>improvement</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>auroc</th>
      <td>0.5340</td>
      <td>0.7880</td>
      <td>0.2541</td>
    </tr>
    <tr>
      <th>f1_mac</th>
      <td>0.1569</td>
      <td>0.5024</td>
      <td>0.3455</td>
    </tr>
    <tr>
      <th>f1_mic</th>
      <td>0.6456</td>
      <td>0.7684</td>
      <td>0.1228</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.7370</td>
      <td>0.7684</td>
      <td>0.0314</td>
    </tr>
  </tbody>
</table>


</div>




```python
# rate
compare_full_rate = pd.DataFrame({
    'baseline_full':[auroc_rating_baseline_full, f1_mac_rating_baseline_full, f1_mic_rating_baseline_full, accuracy_rating_baseline_full],
    'CBOW_full':[auroc_rating_CBOW_full, f1_mac_rating_CBOW_full, f1_mic_rating_CBOW_full, accuracy_rating_CBOW_full]
},index = ['auroc','f1_mac','f1_mic','accuracy'])
compare_full_rate['improvement'] =  (compare_full_rate['CBOW_full']-compare_full_rate['baseline_full'])
compare_full_rate.round(4)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }




    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }

</style>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>baseline_full</th>
      <th>CBOW_full</th>
      <th>improvement</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>auroc</th>
      <td>0.5340</td>
      <td>0.8208</td>
      <td>0.2868</td>
    </tr>
    <tr>
      <th>f1_mac</th>
      <td>0.1569</td>
      <td>0.2780</td>
      <td>0.1211</td>
    </tr>
    <tr>
      <th>f1_mic</th>
      <td>0.6456</td>
      <td>0.6740</td>
      <td>0.0284</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.6456</td>
      <td>0.6740</td>
      <td>0.0284</td>
    </tr>
  </tbody>
</table>


</div>



# Recommand System

### This part is for exist customers, when you input your id, we can output some recommend items for you.


```python
usersPerItem = defaultdict(set)
itemsPerUser = defaultdict(set)
catPerItem = defaultdict(set)
rentforPerItem = defaultdict(set)

for d in dataset:
    user,item,category,rentfor = d['user_id'], d['item_id'],d['category'],d['rented for']
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    catPerItem[item].add(category)
    rentforPerItem[item].add(rentfor)

def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    return numer / denom
```


```python
def user_fav(user_id) :
    ratings = []
    items = []
    rating = []
    for d in dataset:
        if (d['user_id'] == user_id) :
            rating.append(d['rating'])
    max_rating = max(rating)
    for d in dataset:
        if (d['user_id'] == user_id) & (d['rating'] == max_rating):
            items.append(str(d['item_id']))
    items.sort()
    return items[0]

def mostSimilarItem(i): 
    similarities = []
    users = usersPerItem[i]
    candidateItem = set()
    for u in users: 
        candidateItem = candidateItem.union(itemsPerUser[u])
    for i2 in candidateItem: 
        if i2 == i: continue
        sim = Jaccard(users, usersPerItem[i2])
        similarities.append((sim, i2))
    sortedList = [i[1] for i in sorted(similarities, key = lambda y: (-y[0], y[1]))[:10]]
    cats = [catPerItem[j] for j in sortedList]
    occ = [rentforPerItem[j] for j in sortedList]
    result = pd.DataFrame({'itemid':sortedList, 'category': cats, 'occasion': occ})
    return result
```

# Interactive step


```python
UserId = input("Please input your user id: " )
print(mostSimilarItem(user_fav(str(UserId))))
```

    Please input your user id:  420272


        itemid    category                                           occasion
    0  2732613     {shirt}               {date, everyday, party, other, work}
    1   925104     {dress}        {wedding, everyday, party, vacation, other}
    2  1006590     {dress}                             {date, party, wedding}
    3   940419     {dress}  {date, wedding, everyday, party, formal affair...
    4   962489     {dress}     {date, everyday, party, vacation, other, work}
    5  1601603     {dress}       {date, wedding, party, formal affair, other}
    6  2375866     {skirt}                        {vacation, party, everyday}
    7  2363191  {jumpsuit}  {date, everyday, party, formal affair, vacatio...
    8  1635675     {dress}  {date, wedding, everyday, party, formal affair...
    9  2211789    {romper}                 {vacation, party, everyday, other}




## -------------------------------------------------

## Attached Files

`renttherunway_final_data.json`

The raw data collected from RentTheRunway.com, containing 192,544 records from 105,508 users regarding 5,850 items.

To download this file, please go to https://www.dropbox.com/s/9omw6s9bh8vxkgu/renttherunway_final_data.json?dl=0.

`glove.6B.100d.txt`

The corpus which is used to train the Glove model (with the embedding dimensions of 100).

To download this file, please go to https://www.dropbox.com/s/sc5uzmrcype7udr/glove.6B.100d.txt?dl=0 or refer to https://nlp.stanford.edu/projects/glove/.

`code.ipynb`

The code containing all the works mentioned in our report.

`report.pdf`

The final report which includes detailed process.

## Reference

Rishabh Misra, Mengting Wan, Julian McAuley (2018). *Decomposing fit semantics for product size recommendation in metric spaces.*

https://cseweb.ucsd.edu//~jmcauley/pdfs/recsys18e.pdf

Hsin-Yu Chen, Huang-Chin Yen, Tian Tian (2020). *Female Clothing E-commerce Review and Rating.*

https://github.com/hsinyuchen1017/Python-Prediction-of-Rating-using-Review-Text/blob/master/Female%20Clothing%20E-commerce%20Review%20and%20Rating.pdf
