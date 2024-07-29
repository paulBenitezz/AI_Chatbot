import numpy as np
import re
from nltk import download
from nltk.data import find
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords, wordnet
from sklearn.feature_extraction.text import TfidfVectorizer


#Download pre-existing datasets
def check_and_download(resource):
    try:
        find(f'corpora/{resource}')
        print(f"'{resource}' is already downloaded.")
    except LookupError:
        print(f"'{resource}' not found. Downloading now...")
        download(resource)
        print(f"'{resource}' downloaded successfully.")

check_and_download('wordnet')
check_and_download('stopwords')

#Initialize lemmatizer and stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

#Load words
stopWords = set(stopwords.words('english'))

#Tokenization
def tokenize(input):
    # If the input is a string, split it into words
    if isinstance(input, str):
        words = re.findall(r'\b\w+\b', input.lower())  # Improved tokenization using regex
    # If the input is already a list, assume it's already tokenized
    elif isinstance(input, list):
        words = input
    else:
        raise ValueError("Input must be a string or a list of strings")
    
    return words

#Stop Word Removal
def removeStopWords(words):
    return [word for word in words if word not in stopWords]

#Custom Stopwords
def add_custom_stopwords(custom_stopwords):
    global stopWords
    stopWords.update(custom_stopwords)

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
    print(f"BAG OF WORDS: {bow}")
    return bow

# TF-IDF Vectorization
def vectorize_texts(texts):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)
    return vectors, vectorizer

def vectorize(tokens):
    bow = bagOfWords(tokens)
    vector = np.array(list(bow.values()))
    return vector

# Function to clean text
def clean_text(text):
    if isinstance(text, np.ndarray):
        text = ' '.join(map(str, text))
    text = text.lower()  # Lowercase text
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#','', text)  # Remove @mentions and hashtags
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

