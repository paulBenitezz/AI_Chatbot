import tensorflow as tf
import NLP
import json
import os
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential # type: ignore
from keras.layers import Dense, Dropout, SimpleRNN, Input # type: ignore
from keras.optimizers import Adam # type: ignore
from keras.layers import Dense, Dropout, Input # type: ignore
from keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore

versionFile = r'Data/Logs/version_log.json'

#Get latest model version
def get_latest_model_version():
    try:
        with open('Models/model_versions.json', 'r') as f:
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
    #Must import here to avoid circular import
    from Preprocessing import preprocess_input

    processed_texts = [preprocess_input(text) for text in texts]
    vectors, vectorizer = NLP.vectorize_texts(processed_texts)
    return vectors, vectorizer

#Build a new model
def build_model(input_dim, num_classes):
    #Create new model version
    version = get_latest_model_version()
    version += 1
    write_new_version(version)
    
    #Build model architecture
    model = Sequential([
    Input(shape=(input_dim,)),  # Adjust input shape to (timesteps, features)
    SimpleRNN(256, activation='relu', return_sequences=True),
    Dropout(0.5),
    SimpleRNN(128, activation='relu'),
    Dropout(0.3),
    # Dense(256, activation='relu'),  # Additional dense layer
    # Dense(128, activation='relu'),  # Another additional dense layer
    # Dense(128, activation='relu'),  # Another additional dense layer
    Dense((num_classes), activation='softmax')
    ])

    #Compile and save new model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    save_model(model, f'Models/model_version{version}.keras')
    return model

#Train a new or pre-existing model -- ALL TRAINING THUS FAR HAS SEVERELY UNDERFITTED DATA (REALLY LOW ACCURACY, REALLY HIGH LOSS)
def train_model(xTrain, yTrain, xVal, yVal, batch_size=32, epochs=50):
    # Encode labels
    label_encoder = LabelEncoder()
    encoded_tags = label_encoder.fit_transform(yTrain)

    # Reshape vectors to 3D (batch_size, timesteps, features)
    xTrain = xTrain.reshape((xTrain.shape[0], 1, xTrain.shape[1]))
    xVal = xVal.reshape((xVal.shape[0], 1, xVal.shape[1]))

    # Model architecture
    model = Sequential([
    Input(shape=(1,xTrain.shape[2])),  # Adjust input shape to (timesteps, features)
    SimpleRNN(256, activation='relu', return_sequences=True),
    Dropout(0.5),
    SimpleRNN(128, activation='relu'),
    Dropout(0.3),
    # Dense(256, activation='relu'),  # Additional dense layer
    # Dense(128, activation='relu'),  # Another additional dense layer
    # Dense(128, activation='relu'),  # Another additional dense layer
    Dense((len(yTrain)), activation='softmax')
    ])
    
    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Early Stopping -- If the model's loss hasn't improved in 3 epochs, stop training
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Model Checkpoint
    checkpoint = ModelCheckpoint(f'Weights/model_version{get_latest_model_version()}.keras', monitor='val_accuracy', save_best_only=True)
    
    # Train model
    history = model.fit(xTrain, yTrain, batch_size=batch_size, epochs=epochs, validation_data=(xVal, yVal), callbacks=[earlyStopping, checkpoint])
    
    # Extract the last epoch's validation accuracy and loss
    val_accuracy = history.history['val_accuracy'][-1]
    val_loss = history.history['val_loss'][-1]

    # Define your criteria for saving the model
    accuracy_threshold = 0.80
    loss_threshold = 0.5

    # Conditional statement to save the model
    if val_accuracy >= accuracy_threshold and val_loss <= loss_threshold:
        save_model(model, f'Models/model_version{get_latest_model_version() + 1}({val_accuracy} accurate).keras')
        print(f"Model saved with val_accuracy: {val_accuracy} and val_loss: {val_loss}")
    else:
        print(f"Model not saved. val_accuracy: {val_accuracy}, val_loss: {val_loss}")

    return model, checkpoint, history

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