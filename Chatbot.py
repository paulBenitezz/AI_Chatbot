# Import necessary library
import sys
from Preprocessing import process_input_to_find_answer

def advisor():
    #Welcome Message
    print("Welcome to the console-based chatbot. Type 'exit' to quit.")


    while True:
        # Get user input
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break
        
        #Process input to generate answer
        response = process_input_to_find_answer(user_input)
        #response = "I'm an idiot."

        #Print response
        print(f"Chatbot: {response}")

#Main function
if __name__ == '__main__':
    advisor()