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
# print(md.head(20))

md=md.sort_values('impe1_score', ascending=False)
# print(md.head(20))

sd = pd.read_csv('data/song_id.csv')
# print(sd.head(20))

user_data_df=pd.read_csv('data/user_data.csv')
user_df = user_data_df.drop(['id', 'sdvx_id', 'name', 'volforce'], axis=1)

print(user_df)

songs_corr = user_df.corr(method='pearson')
songs_corr.to_csv('data/song_pearson_correlation.csv', encoding='utf-8-sig')

# songs_corr_kendall = user_df.corr(method='kendall')
# songs_corr_kendall.to_csv('data/song_kendall_correlation.csv', encoding='utf-8-sig')

# songs_corr_spearman = user_df.corr(method='spearman')
# songs_corr_spearman.to_csv('data/song_spearman_correlation.csv', encoding='utf-8-sig')

corr_dict = dict()

song_name_list = []
for song_name in songs_corr:
    song_name_list.append(song_name)

for song in song_name_list:
    song_col = songs_corr[song]
    song_dict = dict()
    for song_2 in song_name_list:
        song_dict[song_2]=song_col[song_2]
    corr_dict[song]=song_dict

def get_expected_score(user_data:dict, expect_song:str):
    num=0
    deno=0
    val_list=[]
    for song in song_name_list:
        if song==expect_song:continue
        if song in user_data:
            val = corr_dict[song][expect_song]
            val_list.append(val)
    
    val_list.sort()
    std = val_list[90]
    for song in song_name_list:
        if song in user_data:
            val=corr_dict[song][expect_song]
            if song==expect_song: val*=2
            if std>val:continue
            num+=(val-0.5)*user_data[song]
            deno+=(val-0.5)
    return int(num/deno)

my_data = {
    '666': 9730185, 'ΣgØ': 9748294, '* Erm, could it be a Spatiotemporal ShockWAVE Syndrome...?': 9334198, 'MAYHEM': 9918566, 
    'VVelcome!!': 9529593, 'ΣmbryØ': 9662797, '*Feels Seasickness...*': 9730660, 'FIN4LE ～終止線の彼方へ～': 9591006, 'HE4VEN ～天国へようこそ～': 9728155, 
    'iLLness LiLin': 9622911, 'WHITEOUT': 9680522, 'Xronièr': 9729678, 'Lachryma《Re:Queen’M》': 9692231, '777': 9593894, 'Bad Elixir': 9463230, 'Chronomia': 9915572, 
    'onslaught -Retaliation of Bahamūt-': 9738683, 'VALKYRIE ASSAULT': 9913122, '9TH5IN': 9881271, 'ANGER of the GOD': 9931342, 'Calamity Tempest': 9695096, 
    'Daisycutter': 9506676, 'ENDYMION': 8711201, 'Ghost Family Living In Graveyard': 9784321, 'Lancelot ～Flame of the Rebellion～': 9645816, 'Lisa-RICCIA': 9913822, 
    'LubedeR': 9731323, 'ΛΛemoria': 9874188, '†:OLPHEUX:†': 9588744, 'OUTERHEΛVEN': 9600170, 'Redshift 2nd Ignition': 9558295, 'ЯeviveR': 9729294, 
    'TENKAICHI ULTIMATE BOSSRUSH MEDLEY': 9816876, 'THE凸GENERATOR': 9639865, 'Trill auf G': 9905691, 'voltississimo': 9841595, 'Xroniàl Xéro': 9418047, 
    'θコトノハθカプセルθ': 9903964, ' 色を喪った街': 9537572, '卑弥呼': 9502579, '飄える翼追い掛けて': 9570294, 'A Lasting Promise': 9791972, 'Absolute Domination': 9572291, 
    'Awakening': 9710332, 'BELOBOG': 9547797, 'Chrono Diver -PENDULUMs-': 9843032, 'Cross Fire': 9904432, 'Deadly force': 9673790, 'Dyscontrolled Galaxy': 9398029, 
    'Elemental Creation': 9914742, 'Failnaught': 9780392, 'Fin.ArcDeaR': 9884473, 'Fly Like You': 9790141, 'FREEDOM DiVE': 9860644, 'GERBERA-For Finalists-': 9611916, 
    'GODHEART': 9909057, 'Immortal saga': 9926225, 'KAC 2013 ULTIMATE MEDLEY -HISTORIA SOUND VOLTEX- Empress Side': 9524336, 'Last Resort': 9879275, 'Sailing Force': 9845512, 
    'Staring at star': 9906561, 'TWO-TORIAL': 9836776, 'Xéroa': 9623629, 'セイレーン ～悲壮の竪琴～': 9805335, '逆月': 9947066, 'INF-B《L-aste-R》': 9706737, 'FLOWER': 9898232, 
    'Got more raves？': 9908116, 'きたさいたま2000': 9912672, '極圏': 9904684, 'ΑΩ': 9841017, 'Blastix Riotz': 9512466, 'DIABLOSIS::Nāga': 9901751, 'Everlasting Message': 9904381, 
    'FLügeL《Λrp:ΣggyØ》': 9542906, 'KAC 2013 ULTIMATE MEDLEY -HISTORIA SOUND VOLTEX- Emperor Side': 9647058, 'Lord=Crossight': 9915924, 'Preserved Valkyria': 9818580, 
    'XyHATTE': 9797919, '月光乱舞': 9906422, 'Innocent Tempest': 9839650, 'IX': 9911682, 'ラクガキスト': 9916700, '音楽 -resolve-': 9904761, 'Booths of Fighters': 9855617, 
    'For UltraPlayers': 9784927, 'INSECTICIDE': 9868475, 'Quietus Ray': 9828238, 'Verse IV': 9772191, 'バンブーソード・ガール': 9869379, "Bangin' Burst": 9722945, 
    'Black Emperor': 9732943, 'BLACK or WHITE?': 9890862, 'Growth Memories': 9906488, 'snow storm -euphoria-': 9919750, 'XROSS INFECTION': 9793529, '大宇宙ステージ': 9665482, 
    'Ganymede kamome mix': 9848912
    }

expected_data = dict()
for song in song_name_list:
    expected_data[song]=get_expected_score(my_data, song)

for song in expected_data:
    print(song)
    if song in my_data:
        print(my_data[song], end=' ')
    else: print('       ',end=' ')
    print(expected_data[song])