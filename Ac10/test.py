import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

document = "I was told that you went upstairs yesterday, so I walked across the street today.";

words = nltk.word_tokenize(document)

for word in words:
    print(word, nltk.pos_tag([word]))