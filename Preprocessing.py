import NLP
from NLP import TfidfVectorizer
from ML import load_model, get_latest_model_version
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.text import Tokenizer #type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences as pad #type: ignore

#Group member file paths
#filePath  = r"D:/Github/AI_Chatbot/Data/Data/"
filePath = r"C:/Users/green/AI_Chatbot/Data/Data/"

#Files
#fileName = "Test Dataset.json" ---- Original Test File
fileName = "Dataset.json"

file = filePath + fileName

# Assuming existing functions: tokenize, removeStopWords, lemmatize, bagOfWords

def preprocess_input(input_text):
    #Changes list to a string if input_text is a list
    if isinstance(input_text, list):
        input_text = ' '.join(input_text)
    
    #Clean, tokenize, and process input text
    input_text = NLP.clean_text(input_text)
    tokens = NLP.tokenize(input_text)
    tokens = NLP.removeStopWords(tokens)
    processed_tokens = [NLP.stem(token) if NLP.posTag(token) in ['VERB', 'ADV'] else NLP.lemmatize(token) for token in tokens]

    return ' '.join(processed_tokens)

def preprocess_and_vectorize_array(texts, max_features, maxlen):
    from Preprocessing import preprocess_input
    # Preprocess the texts
    #processed_texts = [preprocess_input(text) for text in texts]
    
    # Vectorize the texts
    tokenizer = Tokenizer(num_words=max_features)
    #sequences = tokenizer.texts_to_sequences(processed_texts)
    padded_sequences = pad(texts, maxlen=maxlen)
    return padded_sequences, tokenizer

def load_test_set(file):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            test_set = json.load(f)
        return test_set
    except Exception as e:
        print(f"Error loading file: {e}")
        return []

def pad_or_truncate(vector, target_length):
    vector = list(vector)
    if len(vector) < target_length:
        return vector + [0] * (target_length - len(vector))
    else:
        return vector[:target_length]

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

        similarity = cosine_similarity(input_vector, vector_2d)
        similarity_score = similarity[0][0]

        if similarity_score > highest_similarity:
            highest_similarity = similarity_score
            most_similar_vector = vector
            most_similar_index = idx

    if highest_similarity > similarity_threshold and most_similar_index != -1:
        return most_similar_vector, most_similar_index
    else:
        return None, None

def process_input_to_find_answer(input_text, model_version=get_latest_model_version(), vectorizer=None):
    if vectorizer is None:
        vectorizer = TfidfVectorizer()

    test_set = load_test_set(file)
    if 'faq' not in test_set or not isinstance(test_set['faq'], list):
        return "Error: 'faq' key not found or not in the expected format in the test set."

    faq_entries = test_set['faq']
    tokenized_questions = [preprocess_input(entry['question']) for entry in faq_entries]

    #Fit the vectorizer with the tokenized questions
    vectorizer.fit(tokenized_questions)

    #Process and vectorize input ---- Maybe add this block to new function?
    preprocessed_input = preprocess_input(input_text)
    input_vector = vectorizer.transform([preprocessed_input]).toarray()
    vectors = vectorizer.transform(tokenized_questions).toarray()

    #Similarities always comes out less than 0.5 for some reason, should we change the threshold or strictly use ML option for all predicitons
    similarities = cosine_similarity(input_vector, vectors).flatten()
    most_similar_idx = similarities.argmax()

    if similarities[most_similar_idx] > 0.5:
        return faq_entries[most_similar_idx]['answer']
    else:
        #Load model
        num_classes = len(set([entry['tag'] for entry in faq_entries]))
        model = load_model(model_version, vectors.shape[1], num_classes)
        
        #Make predictions
        predictions = model.predict(input_vector)
        predicted_class = np.argmax(predictions, axis=1)[0]
        
        return faq_entries[predicted_class]['answer']




#------------old version of this function ... still working on the similarity stuff------------
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