import json
import ML

version = ML.get_latest_model_version()

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

if __name__ == "__main__":
    #Get user input to create new model to train or train existing model
    print("Select an option:\n\
        1. Train a new model\n\
        2. Use an existing model")
    userInput = input()

    #If user wants to train existing model enter version number
    if userInput == "2":
        version = input("Enter the model version number you wish to train: ")

    #Load data and train model
    questions, tags, answers = load_data(r'Data/Data/Dataset.json')
    model, vectorizer = ML.train_model(questions, tags)
