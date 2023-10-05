import speech_recognition as sr
import pyttsx3
import datetime
import webbrowser
import os

# Initialize the speech recognition engine
recognizer = sr.Recognizer()

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Function to speak text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Function to listen to user's voice command
def listen():
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio)
            print("You said: " + command)
            return command.lower()
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand your command.")
            return ""
        except sr.RequestError as e:
            print("Could not request results. Check your internet connection.")
            return ""

# Function to execute commands
def execute_command(command):
    if "hello" in command:
        speak("Hello! How can I help you?")
    elif "time" in command:
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        speak("The current time is " + current_time)
    elif "open website" in command:
        speak("Sure, which website would you like to open?")
        website = listen()
        if website:
            url = "https://www." + website + ".com"
            webbrowser.open(url)
    elif "exit" in command:
        speak("Goodbye!")
        exit()
    else:
        speak("I'm sorry, I don't know how to do that.")

# Main loop
while True:
    command = listen()
    if command:
        execute_command(command)

