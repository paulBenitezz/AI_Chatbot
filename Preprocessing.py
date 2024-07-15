import re
import NLP
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

#filePath  = r"D:/Github/AI_Chatbot/Data/Data/"
filePath = r"C:/Users/green/AI_Chatbot/Data/Data/"
fileName = "Test Dataset.json"
file = filePath + fileName

# Assuming existing functions: tokenize, removeStopWords, lemmatize, bagOfWords

def preprocess_input(input_text):
    input_text = clean_text(input_text)
    tokens = NLP.tokenize(input_text)
    tokens = NLP.removeStopWords(tokens)
    processed_tokens = []
    for token in tokens:
        pos = NLP.posTag(token)
        if pos in ['VERB', 'ADV']:
            processed_tokens.append(NLP.stem(token))
        else:
            processed_tokens.append(NLP.lemmatize(token))
    return ' '.join(processed_tokens)

# Vectorize text using TF-IDF
def vectorize_texts(texts):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts)
    return vectors, vectorizer

# Function to clean text
def clean_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'\@\w+|\#','', text)  # Remove @mentions and hashtags
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Load test set from file
def load_test_set(file):
    try:
        with open(file, 'r', encoding='utf-8') as f:  # Added encoding to ensure compatibility
            test_set = json.load(f)
        return test_set
    except FileNotFoundError:
        print(f"File not found: {file}")
        return []
    except json.JSONDecodeError:
        print(f"Failed to decode JSON from the file: {file}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return []

# Pad or truncate vectors to target length
def pad_or_truncate(vector, target_length):
    vector = list(vector)  # Ensure the vector is a list
    if len(vector) < target_length:
        return vector + [0] * (target_length - len(vector))
    else:
        return vector[:target_length]

# Find most similar vector in the test set
def find_most_similar(input_vector, test_set, similarity_threshold=0.5):
    highest_similarity = -1
    most_similar_vector = None
    most_similar_index = -1

    max_length = max(len(vector) for vector in test_set)
    input_vector = pad_or_truncate(input_vector, max_length)
    input_vector = np.array(input_vector).reshape(1, -1)

    for idx, vector in enumerate(test_set):
        vector = pad_or_truncate(vector, max_length)
        vector_2d = np.array(vector).reshape(1, -1)

        if input_vector.shape[1] != vector_2d.shape[1]:
            raise ValueError(f"Incompatible dimension for input_vector and vector: {input_vector.shape[1]} != {vector_2d.shape[1]}")

        similarity = cosine_similarity(input_vector, vector_2d)
        print(f"Cosine Similarity\nInput vector: {input_vector} // Vector 2d: {vector_2d}")
        similarity_score = similarity[0][0]

        
        print(f"Similarity with vector {idx}: {similarity_score}\n")
        if similarity_score > highest_similarity:
            highest_similarity = similarity_score
            most_similar_vector = vector
            most_similar_index = idx

    if highest_similarity > similarity_threshold and most_similar_index != -1:
        return most_similar_vector, most_similar_index
    else:
        return None, None

def process_input_to_find_answer(input_text):
    testDataSet = file
    preprocessed_input = preprocess_input(input_text)
    print(f"Preprocessed input: {preprocessed_input}")
    input_vector = NLP.vectorize(preprocessed_input)
    print(f"input_vector: {input_vector}")

    test_set = load_test_set(testDataSet)
    
    if 'faq' not in test_set or not isinstance(test_set['faq'], list):
        return "Error: 'faq' key not found or not in the expected format in the test set."

    # checking only faq entries for now :p
    faq_entries = test_set['faq']
    tokenized_questions = [preprocess_input(entry['question']) for entry in faq_entries]
    
    vectors, vectorizer = vectorize_texts(tokenized_questions)
    input_vector = vectorizer.transform([preprocessed_input])
    
    similarities = cosine_similarity(input_vector, vectors).flatten()
    most_similar_idx = similarities.argmax()

    if similarities[most_similar_idx] > 0.5:
        return faq_entries[most_similar_idx]['answer']
    else:
        return "Sorry, I don't have an answer for that question."


    # old version of this function ... still working on the similarity stuff
'''        
    #Vectorize keys in test set
    tokenized_keys = [preprocess_input(key) for entry in test_set for key in entry.keys()]
    vectorized_keys = [NLP.vectorize(tokens) for tokens in tokenized_keys]

    
    if not test_set:
        return "Error processing the test set."
    answer = find_most_similar(input_vector, vectorized_keys)
    for entry in test_set:
        for key, value in entry.items():
            tokenkey = preprocess_input(key)
            vectorizedkey = NLP.vectorize(tokenkey)
            vectorizedkey = pad_or_truncate(vectorizedkey, len(answer))
            tokenized_value = preprocess_input(value)
            vectorized_value = NLP.vectorize(tokenized_value)
            print(f"VALUE: {key}\nTOKENIZED KEY: {tokenkey}\nVECTORIZED KEY: {vectorizedkey}\n\n")
            if np.array_equal(answer, vectorized_value):
                answer = test_set[key]
    return answer
'''