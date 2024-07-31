import json
import ML
import Preprocessing
import numpy as np
from tensorflow.keras.datasets import imdb #type: ignore
from sklearn.model_selection import train_test_split as tts
from tensorflow.keras.preprocessing.sequence import pad_sequences as pad #type: ignore

# Load the IMDB dataset
max_features = 10000  # Example value, adjust as needed
maxlen = 500  # Example value, adjust as needed

(xTrain, yTrain), (xTest, yTest) = imdb.load_data(num_words=max_features)
xTrain, xVal, yTrain, yVal = tts(xTrain, yTrain, test_size=0.3, random_state=42)

latestVersion = ML.get_latest_model_version()

def load_data(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
        questions = []
        tags = []
        answers = {}
        
        #Load questions, tags, and answers from dataset
        for item in data['faq']:
            tag = item['tag']
            for question in item['question']:
                questions.append(question)
                tags.append(tag)
            answers[tag] = item['answer']
    
    return questions, tags, answers

def convert_sequences_to_texts(sequences, reverse_word_index):
    return [' '.join([reverse_word_index.get(i - 3, '?') for i in seq]) for seq in sequences]

if __name__ == "__main__":
    xTrain, tokenizer = Preprocessing.preprocess_and_vectorize_array(xTrain, max_features, maxlen)
    xVal, _ = Preprocessing.preprocess_and_vectorize_array(xVal, max_features, maxlen)
    xTest, _ = Preprocessing.preprocess_and_vectorize_array(xTest, max_features, maxlen)

    # Ensure yTest is a numpy array
    yTest = np.array(yTest)
    
    # Get user input to create new model to train or train existing model
    print("Select an option:\n\
        1. Train a new model\n\
        2. Use an existing model")
    userInput = input()

    # If user wants to train existing model enter version number
    if userInput == "2":
        version = input("Enter the model version number you wish to train: ")
        model = ML.load_model(version)
        model, modelCheckpoint, history, timesteps = ML.train_model(xTrain, yTrain, xVal, yVal, batch_size = 32, epochs = 10, model = model, version = version)
    else:
        # Now you can train your model
        model, modelCheckpoint, history, timesteps = ML.train_model(xTrain, yTrain, xVal, yVal, batch_size = 32, epochs = 50)

    # Reshape xTest to match the input shape (batch_size, timesteps, features)
    xTest = np.array([np.pad(seq, (0, timesteps - len(seq))) for seq in xTest])
    model.evaluate(xTest, yTest, batch_size=32, callbacks=[modelCheckpoint])