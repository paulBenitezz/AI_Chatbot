import numpy as np
import string

from nltk import download
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords, wordnet


#Download pre-existing datasets
download('wordnet')
download('stopwords')

# Initialize lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

#Load words
stopWords = set(stopwords.words('english'))

#Tokenization
def tokenize(input):
    
    # If the input is a string, split it into words
    if isinstance(input, str):
        words = input.split(' ')
    # If the input is already a list, assume it's already tokenized
    elif isinstance(input, list):
        words = input
    else:
        raise ValueError("Input must be a string or a list of strings")
    
    processed_words = []
    for word in words:
        if word.isalpha():
            processed_words.append(word.lower())
        else:
            # Remove punctuation from the end of the word
            word = word.rstrip(string.punctuation)
            # If the word is now alphabetic, add it to the list
            if word.isalpha():
                processed_words.append(word.lower())

    return processed_words
    
#Stop Word Removal
def removeStopWords(words):
    return [word for word in words if word not in stopWords]

#Lemmatization
def lemmatize(word):
    pos = posTag(word)
    return lemmatizer.lemmatize(word, pos)
        
#Stemming
def stem(word):
    pos = posTag(word)
    return stemmer.stem(word, pos)

#Part of Speech Tagging
def posTag(word):
    tag = wordnet.synsets(word)
    if not tag:
        return wordnet.NOUN  # Default to noun if POS tag is not found
    pos = tag[0].pos()
    pos_map = {'n': wordnet.NOUN, 'v': wordnet.VERB, 'a': wordnet.ADJ, 'r': wordnet.ADV}
    return pos_map.get(pos, wordnet.NOUN)

#Bag of Words
def bagOfWords(words):
    bow = {}
    words = tokenize(words)

    for word in words:
        pos = posTag(word)

        if pos == 'verb':
            word = stem(word)
        else:
            word = lemmatize(word)

        if word in bow:
            bow[word] += 1
        else:
            bow[word] = 1
    #print(f"BAG OF WORDS: {bow}")
    return bow

def vectorize(tokens):
    bow = bagOfWords(tokens)
    vector = np.array(list(bow.values()))
    return vector