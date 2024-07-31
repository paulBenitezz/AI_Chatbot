import tensorflow as tf
import NLP
import numpy as np
import json
import os
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences as pad # type: ignore
from keras.models import Sequential # type: ignore
from keras.layers import Dense, Dropout, BatchNormalization, Input # type: ignore
from keras.optimizers import Adam # type: ignore
from tensorflow.keras.layers import Input, Embedding, LSTM, Dropout, Dense # type: ignore
from keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore

versionFile = r'Data/Logs/version_log.json'


#Get latest model version
def get_latest_model_version():
    try:
        with open('Data/Logs/version_log.json', 'r') as file:
            lines = file.readlines()
            if not lines:
                return 0
            last_line = lines[-1].strip()
            return json.loads(last_line).get('Version', 0)
    except (json.JSONDecodeError, IndexError) as e:
        print(f"Error reading model version: {e}")
        return 0

#Save new model version
def write_new_version(version):
    with open(versionFile, 'a') as file:
        version_info = "\n{"+f"\"Version\": {version}"+"}"
        file.write(version_info)

#Build a new model
def build_model(xTrain, yTrain, embedding_dim = 100, timesteps = None, vocab_size = None):
    if timesteps is None:
        timesteps = max(len(seq) for seq in xTrain)
    if vocab_size is None:
        vocab_size = len(set(word for seq in xTrain for word in seq))

    #Create new model version
    version = get_latest_model_version()
    version += 1
    
    #Build model architecture
    # Model architecture
    model = Sequential([
        Input(shape=(timesteps,)),  # Adjust input shape to (timesteps,)
        Embedding(input_dim=vocab_size + 1, output_dim=embedding_dim, input_length=timesteps),
        LSTM(64, return_sequences=False),
        Dropout(0.1), #Adjust or remove this if not good
        Dense(128, activation='relu'),
        Dropout(0.1), #0.4 so far is best dropout rate for this spot, maybe try removing this one and adjusting the previous one
        BatchNormalization(), #Remove if not good
        Dense(len(set(yTrain)), activation='softmax')
    ])

    return model, version
'''LR = 0.007 no Dropout or batch normalization had been best overall version (VERSION 2). ***NEED TO TRY MORE EPOCHS AND/OR ADJ BATCH SIZE***'''

#Train a new or pre-existing model -- ALL TRAINING THUS FAR HAS SEVERELY UNDERFITTED DATA (REALLY LOW ACCURACY, REALLY HIGH LOSS)
def train_model(xTrain, yTrain, xVal, yVal, batch_size=32, epochs=10, embedding_dim=100, learning_rate=0.006, model = None, version = None):
    # Determine the maximum sequence length
    timesteps = max(len(seq) for seq in xTrain)

    # Determine the vocabulary size
    vocab_size = len(set(word for seq in xTrain for word in seq))

    # Encode labels
    label_encoder = LabelEncoder()
    encoded_tags = label_encoder.fit_transform(yTrain)

    # Reshape vectors to 2D (batch_size, timesteps)
    # Reshape vectors to 3D (batch_size, timesteps, features)
    xTrain = np.array([np.pad(seq, (0, timesteps - len(seq))) for seq in xTrain])
    xVal = np.array([np.pad(seq, (0, timesteps - len(seq))) for seq in xVal])

    # Model architecture
    if model is None:
        model, version = build_model(xTrain, yTrain, embedding_dim, timesteps = timesteps, vocab_size = vocab_size)

    # Compile the model
    model.compile(optimizer=Adam(learning_rate = learning_rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Early Stopping -- If the model's loss hasn't improved in 10 epochs, stop training
    earlyStopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    # Model Checkpoint
    checkpoint = ModelCheckpoint(f'Weights/model_version{version}.keras', monitor='val_accuracy', save_best_only=True)

    # Train model
    history = model.fit(xTrain, encoded_tags, batch_size=batch_size, epochs=epochs, validation_data=(xVal, yVal), callbacks=[earlyStopping, checkpoint])

    # Extract the last epoch's validation accuracy and loss
    val_accuracy = history.history['val_accuracy'][-1]
    val_loss = history.history['val_loss'][-1]

    # Define your criteria for saving the model
    accuracy_threshold = 0.80
    loss_threshold = 0.7

    # Conditional statement to save the model
    if val_accuracy >= accuracy_threshold and val_loss <= loss_threshold:
        save_model(model, f'Models/model_version{version}.keras', version)
        print(f"Model saved with validation accuracy of {val_accuracy} and validation loss of {val_loss}")
    else:
        print(f"Model not saved. Validation accuracy: {val_accuracy}, Validation loss: {val_loss}")

    return model, checkpoint, history, timesteps

#Save new model
def save_model(model, file_path, version):
    model.save(file_path)
    write_new_version(version)
    print(f"Model saved to {file_path}")

#Load model
def load_model(version):
    model_path = f'Models/model_version{version}.keras'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        print(f"Model version {version} not found.")
        return None
    
# #Preprocess and vectorize the training data
# def preprocess_and_vectorize(texts):
#     #Must import here to avoid circular import
#     from Preprocessing import preprocess_input

#     processed_texts = [preprocess_input(text) for text in texts]
#     vectors, vectorizer = NLP.vectorize_texts(processed_texts)
#     return vectors, vectorizer