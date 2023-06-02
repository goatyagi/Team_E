#TeamE
# s1290042 Yahagi Takuya 55%
# s1290062 Miyazaki Souta 15%
# s1290067 Urade Rikuto 15%
# s1290073 Kondo Tatsuya 15%

# The project repository's constructure is like following.
    # in /Ac11/  : Data(directory), main_RFver.py, main_knnver.py, run.ipynb
    # in /Ac11/Data : Authordata(directory), Textdata(directry)
    # in /Ac11/Data/Authordata/ : the csv file of author data (author.csv)
    # in /Ac11/Data/Textdata/ : there are many .txt files
    
# The data format of this program.
    # 1. author.csv
        # Each row have one factor that indicate author name.
        # The order of author name is depends on the order of text file in the Textdata directory.
    # 2. .txt
        # the file just include the sentences.
        # ex) I have been to Hokkaido twice. 
    
    
# Step1. Importing data and getting feature from texts.

import nltk

import os
import re
import pandas as pd

# ----------------Importing part-----------------

# Because I use Jupyter notebook, I need to escape the case that there is .ipynb_checkpoint in a Textdata directory.
# So, by using re, files is the name of file that is in Textdata directory, also that end wit .(txt).
files = [ i for i in os.listdir("./Data/Textdata/") if re.match('.*(.txt)$', i)]
files.sort()
document = []

for f in files:
    docs = open("./Data/Textdata/"+f, "r").read()
    document.append(docs)

words = []
Pos = []

# ----------------Tokenize part------------------------------

for i in range(len(document)):
    words.append(nltk.word_tokenize(document[i]))
    Pos.append(nltk.pos_tag(words[i]))

# -----------------Achiving Feature part 1(adverbs)---------------
# In here, by using the list of words, we classify the type of adverbs.
place = ['abroad', 'above', 'below', 'downstairs', 'upstairs', 'far', 'here', 'there', 'home', 'near', 'nowhere', 'anywhere', 'everywhere', 'outside', 'inside', 'under', 'up', 'across', 'around', 'away', 'beside', 'beyond', ]
time = ['now', 'today', 'tomorrow', 'yesterday', 'recently', 'lately', 'soon', 'early', 'already', 'then', 'still', 'yet', 'later', 'immediately', 'finally', 'before','after', 'afterwards']


P = []
T = []

for i in range(len(document)):
    place_ad =[]
    time_ad = []
    
    for factor in words[i]:
        if factor.lower() in place:
            place_ad.append(factor.lower())
        elif factor.lower() in time:
            time_ad.append(factor.lower())
                    
    P.append(len(place_ad))
    T.append(len(time_ad))

# -----------------Achiving Feature part 2(verbs) -------------------

# In here, I classify each verbs by judging whether the verbs that is next have, has, had is past participle form.

# Attention: In the case sentence includes "have had", the pos_tag of 'had' is not 'VBN'. So, the condition of perfect tense is true if the sentence include following pattern.
    # 1. have/has/had + had
    # 2. have/has/had + VBN
PerfectV = []
NonPerfectV = []
Num_words = []

for i in range(len(document)):    
    
    v_p = []
    v_np = []
    flag = 0 
    
    for j in range (len(Pos[i])):
        if flag != 0:
            flag -= 1
            continue
            
        if re.match("have|has|had", Pos[i][j][0].lower()):
            if Pos[i][j+1][1] == 'VBN' or Pos[i][j+1][0] == 'had':
                v_p.append(Pos[i][j+1][0])
                if Pos[i][j+1][0] == 'had':
                    flag = 1
            elif Pos[i][j+2][1] == 'VBN' or Pos[i][j+2][0] == 'had':
                v_p.append(Pos[i][j+2][0])
                if Pos[i][j+2][0] == 'had':
                    flag = 2
            else:
                v_np.append(Pos[i][j][0])
        elif Pos[i][j][1].startswith('VB') and Pos[i][j][1] != 'VBN' and Pos[i][j][1] != 'VBG':
            v_np.append(Pos[i][j][0])
    
    PerfectV.append(len(v_p))
    NonPerfectV.append(len(v_np))
    
# -----------------Making DataFrame and make csv file------------------------------

author = pd.read_csv("./Data/Authordata/author.csv", index_col=0)
    
table = pd.DataFrame({"Num_Place":P,
                      "Num_Time" : T}, index=files)
table["Num_ad"] = table["Num_Place"] + table["Num_Time"]

Au = []
for i in range(len(author)):
    Au.append(author['Author'][i])

table["PerfectV"] = PerfectV
table["NonPerfectV"] = NonPerfectV
table["Num_Verbs"] = table["PerfectV"] + table["NonPerfectV"]
table["featureV1"] = (table["Num_Place"]-table["Num_Time"]) / (table["Num_ad"]+1)
table["featureV2"] = (table["PerfectV"]-table["NonPerfectV"]) / (table["Num_Verbs"]+1)
table["Author"] = Au


print(table)
table.to_csv("texts.csv")


# Step2. Training and Verification

import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

#-------------------Dividing dataset into for tain and test-----------------------------

texts = pd.read_csv("texts.csv", index_col = 0)
texts = texts.sample(frac=1, random_state=0) # shuffle the order of dataframe

X = texts.loc[:,['featureV1', 'featureV2']]
Y = []

for i in range(len(texts)):
    Y.append(texts['Author'][i])

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.1,random_state=3)

#------------------Train and test------------------------------------------------------------

rf = RandomForestClassifier()
rf.fit(X_train, Y_train)

Y_pred = rf.predict(X_test)
print("The score of rf classifier is ", rf.score(X_test, Y_test))


# Step3. Trying to predict by new data.


#-----------------Making new data-------------------
sentence = "I had lived in far from Tokyo until yesterday." # If you want to try another sentence, you change here.

# ----------------Tokenize Part--------------------
words2 = nltk.word_tokenize(sentence)
pos2 = nltk.pos_tag(words2)

P2 = []
T2 = []

# ---------------Get Feature 1--------------------
place_ad =[]
time_ad = []
    
for factor in words2:
    if factor.lower() in place:
        place_ad.append(factor.lower())
    elif factor.lower() in time:
        time_ad.append(factor.lower())
                    
P2.append(len(place_ad))
T2.append(len(time_ad))

# -------------Get Feature 2-------------------
v_p2 = []
v_np2 = []
flag = 0

for j in range(len(pos2)):
    if flag != 0:
        flag -= 1
        continue
    if re.match("have|has|had", pos2[j][0].lower()):
        if pos2[j+1][1] == 'VBN' or pos2[j+1][0] == 'had':
            v_p2.append(pos2[j+1][0])
            if pos2[j+1][0] == 'had':
                flag = 1
        elif pos2[j+2][1] == 'VBN' or pos[j+2][0] == 'had':
            v_p2.append(pos2[j+2][0])
            if pos2[j+2][0] == 'had':
                flag = 2
        else:
            v_np2.append(pos2[j][0])
    elif pos2[j][1].startswith('VB') and pos2[j][1] != 'VBN' and pos2[j][1] != 'VBG':
        v_np2.append(pos2[j][0])
        
PV2 = []
PV2.append(len(v_p2))

NPV2 = []
NPV2.append(len(v_np2))

# ------------------Making new table-----------------------------------
table2 = pd.DataFrame({"Num_Place":P2,
                      "Num_Time" : T2})

table2["Num_ad"] = table2["Num_Place"] + table2["Num_Time"]
table2["PerfectV"] = PV2
table2["NonPerfectV"] = NPV2
table2["Num_Verbs"] = table2["PerfectV"] + table2["NonPerfectV"]
table2["featureV1"] = (table2["Num_Place"]-table2["Num_Time"]) / (table2["Num_ad"]+1)
table2["featureV2"] = (table2["PerfectV"]-table2["NonPerfectV"]) / (table2["Num_Verbs"]+1)

print(table2.loc[:,["featureV1","featureV2"]])

#--------------------Predicting by new data-------------------------
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)

pred2_Y = rf.predict(table2.loc[:,["featureV1", "featureV2"]])

print("Program predicate the author of ", sentence, " is ", pred2_Y)
