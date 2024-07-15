# Import necessary library
import sys
from Preprocessing import process_input_to_find_answer

def advisor():
    print("Welcome to the console-based chatbot. Type 'exit' to quit.")
    while True:
        #Get user input
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

        #Bot response
        response = process_input_to_find_answer(user_input)
        #response = "I'm an idiot."

        print(f"Chatbot: {response}")

if __name__ == '__main__':
    advisor()