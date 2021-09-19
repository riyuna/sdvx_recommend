import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from ast import literal_eval
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
# from surprise import Reader, Dataset, SVD, evaluate

import warnings; warnings.simplefilter('ignore')

md = pd.read_csv('data/song_data.csv')

avg_counts_imp = md['avg_score_imp1_1'].astype('int')
avg_counts_crm = md['avg_score_crim1'].astype('int')

def difficulty_crimson(x):
    crim1 = x['avg_score_crim1']
    crim2 = x['avg_score_crim2']
    crim3 = x['avg_score_crim3']
    crim4 = x['avg_score_crim4']
    return (crim1+crim2+crim3+crim4)//4

def difficulty_imperial_1(x):
    impe1_1 = x['avg_score_imp1_1']
    impe1_2 = x['avg_score_imp1_2']
    impe1_3 = x['avg_score_imp1_3']
    impe1_4 = x['avg_score_imp1_4']
    return (impe1_1+impe1_2+impe1_3+impe1_4)//4

md['crim_score'] = md.apply(difficulty_crimson, axis=1)
md['impe1_score'] = md.apply(difficulty_imperial_1, axis=1)
md = md.sort_values('crim_score', ascending=False)
print(md.head(20))

md=md.sort_values('impe1_score', ascending=False)
print(md.head(20))

sd = pd.read_csv('data/song_id.csv')
print(sd.head(20))