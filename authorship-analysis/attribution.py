"""Authorship Attribution

Usage:
  attribution.py --words <filename>
  attribution.py --chars=<n> <filename>
  attribution.py (-h | --help)
  attribution.py --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  --words
  --chars=<n>  Length of char ngram.

"""

import sys
import os
import math
from utils import process_document_words, process_document_ngrams, get_documents, extract_vocab, top_cond_probs_by_author
from docopt import docopt


def count_docs(documents):
    return len(documents)

def count_docs_in_class(documents, c):
    count=0
    for values in documents.values():
        if values[0] == c:
            count+=1
    return count

def concatenate_text_of_all_docs_in_class(documents,c):
    words_in_class = {}
    for d,values in documents.items():
        if values[0] == c:
            words_in_class.update(values[2])
    return words_in_class

def train_naive_bayes(classes, documents):
    vocabulary = extract_vocab(documents)
    conditional_probabilities = {}
    for t in vocabulary:
        conditional_probabilities[t] = {}
    priors = {}
    print("\n\n***\nCalculating priors and conditional probabilities for each class...\n***")
    for c in classes:
         priors[c] = count_docs_in_class(documents,c) / count_docs(documents)
         print("\nPrior for",c,priors[c])
         class_size = count_docs_in_class(documents, c)
         print("In class",c,"we have",class_size,"document(s).")
         words_in_class = concatenate_text_of_all_docs_in_class(documents,c)
         #print(c,words_in_class)
         print("Calculating conditional probabilities for the vocabulary.")
         denominator = sum(words_in_class.values())
         for t in vocabulary:
             if t in words_in_class:
                 conditional_probabilities[t][c] = (words_in_class[t] + alpha) / (denominator * (1 + alpha))
                 #print(t,c,words_in_class[t],denominator,conditional_probabilities[t][c])
             else:
                 conditional_probabilities[t][c] = (0 + alpha) / (denominator * (1 + alpha))
    return vocabulary, priors, conditional_probabilities

def apply_naive_bayes(classes, vocabulary, priors, conditional_probabilities, test_document):
    scores = {}
    if feature_type == "chars":
        author, doc_length, words = process_document_ngrams(test_document,ngram_size)
    elif feature_type == "words":
        author, doc_length, words = process_document_words(test_document)
    for c in classes:
        scores[c] = math.log(priors[c])
        for t in words:
            if t in conditional_probabilities:
                for i in range(words[t]):
                    scores[c] += math.log(conditional_probabilities[t][c])
    print("\n\nNow printing scores in descending order:")
    for author in sorted(scores, key=scores.get, reverse=True):
        print(author,"score:",scores[author])
        
# for verification
def apply_naive_bayes2(classes, vocabulary, priors, conditional_probabilities, test_document):
    scores = {}
    Author = []
    if feature_type == "chars":
        author, doc_length, words = process_document_ngrams(test_document,ngram_size)
    elif feature_type == "words":
        author, doc_length, words = process_document_words(test_document)
    for c in classes:
        scores[c] = math.log(priors[c])
        for t in words:
            if t in conditional_probabilities:
                for i in range(words[t]):
                    scores[c] += math.log(conditional_probabilities[t][c])
    k = 0
    for author in sorted(scores, key=scores.get, reverse=True):
        if scores[author] == scores[("Austen")]:
            k += 1
            print(k, " is written by Austen")
            Author.append("Austen")
        else:
            k += 1
            print(k, " is not wirtten by Austen")
            Author.append("Other")
    return Author

# main part
if __name__ == '__main__':
    arguments = docopt(__doc__, version='Authorship Attribution 1.1')

    if arguments["--words"]:
        feature_type = "words"
        ngram_size = -1
    if arguments["--chars"]:
        feature_type = "chars"
        ngram_size = int(arguments["--chars"])

    testfile = arguments["<filename>"]
    
    alpha = 0.1
    classes = ["Austen", "Carroll", "Grahame", "Shelley"]
    documents = get_documents(feature_type, ngram_size)

    vocabulary, priors, conditional_probabilities = train_naive_bayes(classes, documents)

    for author in classes:
        print("\nBest features for",author)
        top_cond_probs_by_author(conditional_probabilities, author, 10)

    apply_naive_bayes(classes, vocabulary, priors, conditional_probabilities, testfile)
    
# added code ( for verification )

    if arguments["--words"]:
        feature_type = "words"
        ngram_size = -1
    if arguments["--chars"]:
        feature_type = "words"
        ngram_size = -1
    
    file2 = arguments["<filename>"]
    
    documents = get_documents(feature_type, ngram_size)
    
    vocabulary, priors, conditional_probabilities = train_naive_bayes(classes, documents)
    
    for author in classes:
        print("\nBest features for", author)
        top_cond_probs_by_author(conditional_probabilities, author, 10)
        
    Author = apply_naive_bayes2(classes, vocabulary, priors, conditional_probabilities, file2)
    
# Unit 6 Part

import nltk

import os
import re
import pandas as pd

# ----------------Importing part-----------------

# Because I use Jupyter notebook, I need to escape the case that there is .ipynb_checkpoint in a Textdata directory.
# So, by using re, files is the name of file that is in Textdata directory, also that end wit .(txt).

document = []
files = [os.path.join('./data/training/', f) for f in os.listdir('./data/training/') if os.path.isfile(os.path.join('./data/training/', f))]
    
for f in files:
    docs = open(f, "r").read()
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
    
# -----------------Achiving Feature part 3(type vs. token) -------------------

TypeL = []
TokenL = []

for i in range(len(document)):
    typelist = []
    tokenlist = []
    
    for factor in words[i]:
        if factor.lower() not in typelist:
            typelist.append(factor)
        tokenlist.append(factor)

    TypeL.append(len(typelist))
    TokenL.append(len(tokenlist))
    
# -----------------Making DataFrame and make csv file------------------------------

table = pd.DataFrame({"Num_Place":P,
                      "Num_Time" : T}, index=files)
table["Num_ad"] = table["Num_Place"] + table["Num_Time"]


table["PerfectV"] = PerfectV
table["NonPerfectV"] = NonPerfectV
table["Num_Verbs"] = table["PerfectV"] + table["NonPerfectV"]
table["Num_Type"] = TypeL
table["Num_Token"] = TokenL
table["featureV1"] = (table["Num_Place"]-table["Num_Time"]) / (table["Num_ad"]+1)
table["featureV2"] = (table["PerfectV"]-table["NonPerfectV"]) / (table["Num_Verbs"]+1)
table["featureV3"] = table["Num_Type"] / table["Num_Token"]
table["Author"] = Author

print(table)