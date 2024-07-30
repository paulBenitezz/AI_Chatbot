import tensorflow as tf
import Preprocessing
import NLP
import json
import os
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential # type: ignore
from keras.layers import Dense, Dropout # type: ignore
from keras.optimizers import Adam # type: ignore
from keras.layers import Dense, Dropout, Input # type: ignore
from keras.callbacks import EarlyStopping # type: ignore

versionFile = r"D:\Github\AI_Chatbot\Data\Logs\version_log.json"

#Get latest model version
def get_latest_model_version():
    try:
        with open('D:\Github\AI_Chatbot\Models\model_versions.json', 'r') as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1]
                return json.loads(last_line).get('Version', 0)
    except FileNotFoundError:
        return 0
    return 0

#Save new model version
def write_new_version(version):
    with open(versionFile, 'a') as file:
        json.dump({'Version': version}, file)
        file.write('\n')

#Preprocess and vectorize the training data
def preprocess_and_vectorize(texts):
    processed_texts = [Preprocessing.preprocess_input(text) for text in texts]
    vectors, vectorizer = NLP.vectorize_texts(processed_texts)
    return vectors, vectorizer

#Build a new model
def build_model(input_dim, num_classes):
    #Create new model version
    version = get_latest_model_version()
    version += 1
    write_new_version(version)
    
    #Build model architecture
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_dim=input_dim),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    #Compile and save new model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    save_model(model, f'Models/model_version{version}.keras')
    return model

#Train a new or pre-existing model -- ALL TRAINING THUS FAR HAS SEVERELY UNDERFITTED DATA (REALLY LOW ACCURACY, REALLY HIGH LOSS)
def train_model(questions, tags, batch_size=32, epochs=50):
    #Data preprocessing
    vectors, vectorizer = preprocess_and_vectorize(questions)
    
    label_encoder = LabelEncoder()
    encoded_tags = label_encoder.fit_transform(tags)
    
    #Model architecture
    model = Sequential()
    model.add(Input(shape=(vectors.shape[1],)))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(set(tags)), activation='softmax'))
    
    #Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    #Early Stopping -- If the model's loss hasn't improved in 3 epochs, stop training
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    #Train model
    model.fit(vectors.toarray(), encoded_tags, batch_size=batch_size, epochs=epochs, validation_split=0.2, callbacks=[early_stopping])
    
    #Save trained model
    version = get_latest_model_version() + 1
    model.save(f'Models/model_version{version}.keras')
    
    return model, vectorizer

#Save new model
def save_model(model, file_path):
    model.save(file_path)
    print(f"Model saved to {file_path}")

#Load model
def load_model(version, input_dim, num_classes):
    model_path = f'Models/model_version{version}.keras'
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    else:
        # Build a new model if the file does not exist
        model = build_model(input_dim, num_classes)
        model.save(model_path)
        return model