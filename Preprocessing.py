import NLP
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity

# Assuming existing functions: tokenize, removeStopWords, lemmatize, bagOfWords

def preprocess_input(input_text):
    tokens = NLP.tokenize(input_text)
    tokens = NLP.removeStopWords(tokens)
    #print(f"TOKENS:\n{tokens}")
    processed_tokens = []
    for token in tokens:
        pos = NLP.posTag(token)
        # Decide if the token should be stemmed
        # This example stems the token if lemmatization does not change it,
        # but you might use different criteria
        if pos == 'verb' or pos == 'adverb':
            stemmed_token = NLP.stem(token)
            processed_tokens.append(stemmed_token)
        else:
            lemmatized_token = NLP.lemmatize(token)
            processed_tokens.append(lemmatized_token)
    
    return processed_tokens


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

def pad_or_truncate(vector, target_length):
    """Pads or truncates a vector to the target length."""
    vector = list(vector)  # Ensure the vector is a list
    vector_length = len(vector)
    if vector_length < target_length:
        # Pad with zeros
        return vector + [0] * (target_length - vector_length)
    elif vector_length > target_length:
        # Truncate the vector
        return vector[:target_length]
    else:
        return vector

def find_most_similar(input_vector, test_set):
    highest_similarity = -1
    most_similar_vector = None
    
    # Find the maximum length of vectors in the test set
    max_length = max(len(vector) for vector in test_set)
    # Ensure input_vector is the same length
    input_vector = pad_or_truncate(input_vector, max_length)
    input_vector = np.array(input_vector).reshape(1, -1)  # Ensure input_vector is 2D

    for vector in test_set:
        # Pad or truncate the vector to the same length as input_vector
        vector = pad_or_truncate(vector, max_length)
        vector_2d = np.array(vector).reshape(1, -1)
        
        # Check dimensions
        if input_vector.shape[1] != vector_2d.shape[1]:
            raise ValueError(f"Incompatible dimension for input_vector and vector: {input_vector.shape[1]} != {vector_2d.shape[1]}")
        
        similarity = cosine_similarity(input_vector, vector_2d)
        similarity_score = similarity[0][0]
        
        if similarity_score > highest_similarity:
            highest_similarity = similarity_score
            most_similar_vector = vector
            
    return most_similar_vector

def process_input_to_find_answer(input_text):
    testDataSet = r"C:\Users\green\Desktop\UTSA\Summer 2024\Artificial Intelligence\Project\Data Set\Test Dataset.json"
    #Vectorize user input
    preprocessed_input = preprocess_input(input_text)
    input_vector = NLP.vectorize(preprocessed_input)
    #Load test set data
    test_set = load_test_set(testDataSet)
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