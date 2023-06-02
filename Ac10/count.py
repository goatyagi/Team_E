# team member : (the percentage of contribution)
# s1290067 Urade Rikuto  :10
# s1290042 Yahagi Takuya  :70 
# s1290062 Miyazaki Souta  :10
# s1290073  Kondo Tatsuya  : 10

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split

import pandas as pd

import re
import os

files = [ i for i in os.listdir("./Data/") if re.match('.*(.txt)$', i)]
files.sort()

document = []

for f in files:
    docs = open("./Data/"+f, "r").read()
    document.append(docs)
    
words = []

for i in range(len(document)):
       words.append(nltk.word_tokenize(document[i]))

place = ['abroad', 'above', 'below', 'downstairs', 'upstairs', 'far', 'here', 'there', 'home', 'near', 'nowhere', 'anywhere', 'everywhere', 'outside', 'inside', 'under', 'up', 'across', 'around', 'away', 'beside', 'beyond', ]
time = ['now', 'today', 'tomorrow', 'yesterday', 'recently', 'lately', 'soon', 'early', 'already', 'then', 'still', 'yet', 'later', 'immediately', 'finally', 'before','after', 'afterwards']

# We cannot correctly get pos_tag. ( ex. yesterday == NN( noun, singular or math ),  across == IN (preposition).  So, we do not use pos_tag for counting.

P = []
T = []
U = []
for i in range(len(document)):
    place_ad =[]      # place_ad is the list that store the adverbials of place.
    time_ad = []       # time_ad is the list that store the adverbials of time.
    unknown_ad = [] # unknown_ad is the list of adverbials are neither about time nor about place.  
    for factor in words[i]:
            if factor.lower() in place:
                place_ad.append(factor)          
            elif factor.lower() in time:
                time_ad.append(factor)
            else:
                if nltk.pos_tag([factor])[0][1] == 'RB':
                    unknown_ad.append(factor)
    P.append(len(place_ad))
    T.append(len(time_ad))
    U.append(len(unknown_ad))
                
table = pd.DataFrame({"num_p":P,
                      "num_t" :T,
                      "num_u" :U}, index=files)

table["num_words"] = table["num_p"] + table["num_t"] + table["num_u"]

table.to_csv("texts.csv")

texts = pd.read_csv("texts.csv", index_col=0)
print(texts.head())

X = texts
Y = [0, 1, 2, 3, 0]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3,random_state=3)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
Y_pred = knn.predict(X_test)

print(knn.score(X_test, Y_test))

doc2 = "Yesterday, I did my homework soon after the school near the class room."

words2 = nltk.word_tokenize(doc2)
        
P2 = []
T2 = []
U2 = []

place_ad = []
time_ad = []
unknown_ad = []

for factor in words2:
    if factor.lower() in place:
        place_ad.append(factor)          
    elif factor.lower() in time:
        time_ad.append(factor)    
    else:
        if nltk.pos_tag([factor])[0][1] == 'RB':
            unknown_ad.append(factor)
P2.append(len(place_ad))
T2.append(len(time_ad))
U2.append(len(unknown_ad))

table2 = pd.DataFrame({"num_p":P2,
                      "num_t" :T2,
                      "num_u" :U2})

table2["num_words"] = table2["num_p"] + table2["num_t"] + table2["num_u"]

print(knn.predict(table2))
