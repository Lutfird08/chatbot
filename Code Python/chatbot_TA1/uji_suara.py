# import necessary libraries
import warnings
warnings.filterwarnings("ignore")
import nltk # type: ignore
from nltk.stem import WordNetLemmatizer
import json
import pickle
import serial
import webbrowser
from tensorflow import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
from keras.models import load_model

# Inisialisasi koneksi serial dengan Arduino
# arduino = serial.Serial('COM10', 115200)

# load the saved model file
model = load_model('model.h5')
intents = json.loads(open("intents.json").read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

lemmatizer = WordNetLemmatizer()

print(classes)

####
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words) 
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    error = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>error]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    global return_list
    return_list = []
    
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

#####

# function to get the response from the model

def getResponse(ints, intents_json):
    global tag
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

# function to predict the class and get the response
def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)    
    return res

#####

# function to start the chat bot which will continue till the user type 'end'
import speech_recognition as sr

# Your existing code...
'''
def voice():
    r = sr.Recognizer()
    with sr.Microphone(device_index=1) as source:
        r.adjust_for_ambient_noise(source)
        print("Berbicara")
        r.pause_threshold = 1
        audio = r.listen(source)

    try:
        print("Mengolah")
        query = r.recognize_google(audio, language='id')
        print("Kamu Berbicara: " + query)
        return query

    except Exception as e:
        print(e)
        print("Tidak dapat mendengar apapun")
        return "None"
'''
def voice():
    r = sr.Recognizer()
    while True:
        try:
            with sr.Microphone(device_index=1) as source:
                r.adjust_for_ambient_noise(source)
                print("Berbicara")
                r.pause_threshold = 1
                audio = r.listen(source)

            print("Mengolah")
            query = r.recognize_google(audio, language='id')
            print("Kamu Berbicara: " + query)
            return query
        except Exception as e:
            print("Tidak dapat mendengar apapun")
            continue  # Kembali ke blok berbicara

from gtts import gTTS
from io import BytesIO
import pygame

def speak(text):
    pygame.init()
    pygame.mixer.init()
    tts_obj = gTTS(text=text, lang="id")
    mp3_fp = BytesIO()
    tts_obj.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    pygame.mixer.music.load(mp3_fp)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(5)

import datetime
import locale

def start_chat_with_voice():
    print("Bot: Saya asisten mu! Katakan apa yang kamu butuhkan.\n\n")
    while True:
        inp = voice().lower()  # Use voice input instead of text input
        if inp == "end":
            break
        if inp == '' or inp == '*':
            print('Please re-phrase your query!')
            print("-" * 50)
        else:
            response = chatbot_response(inp)
            print(f"Bot: {response}" + '\n')
            speak(response)  # Speak the response using pygame

            # Code for sending commands to Arduino...
            if tag=='lampu1_hidup':
                '''arduino.write('lampu1_hidup'.encode())'''
            elif tag=='lampu1_mati':
                '''arduino.write('lampu1_mati'.encode())'''
            elif tag=='lampu2_hidup':
                '''arduino.write('lampu2_hidup'.encode())'''
            elif tag=='lampu2_mati':
                '''arduino.write('lampu2_mati'.encode())'''
            elif tag=='lampu3_hidup':
                '''arduino.write('lampu3_hidup'.encode())'''
            elif tag=='lampu3_mati':
                '''arduino.write('lampu3_mati'.encode())'''
            
            elif tag=='kipas_hidup':
                '''arduino.write('kipas_hidup'.encode())'''
            elif tag=='kipas_naik':
                '''arduino.write('kipas_naik'.encode())'''
            elif tag=='kipas_turun':
                '''arduino.write('kipas_turun'.encode())'''    
            elif tag=='kipas_mati':
                '''arduino.write('kipas_mati'.encode())'''
            
            elif tag=='ac_hidup':
                '''arduino.write('ac_hidup'.encode())'''
            elif tag=='ac_naik':
                '''arduino.write('ac_naik'.encode())'''
            elif tag=='ac_turun':
                '''arduino.write('ac_turun'.encode())'''
            elif tag=='ac_mati':
                '''arduino.write('ac_mati'.encode())'''

            elif tag=='kunci_hidup':
                '''arduino.write('kunci_hidup'.encode())'''
            elif tag=='kunci_mati':
                '''arduino.write('kunci_mati'.encode())'''

            elif tag=='kran_mati':
                '''arduino.write('kran_mati'.encode())'''
            elif tag=='kran_hidup':
                '''arduino.write('kran_hidup'.encode())'''
          
            elif tag=='tirai_hidup':
                '''arduino.write('tirai_hidup'.encode())'''
            elif tag=='tirai_mati':
                '''arduino.write('tirai_mati'.encode())'''

            elif tag=='pompa_hidup':
                '''arduino.write('pompa_hidup'.encode())'''
            elif tag=='pompa_mati':
                '''arduino.write('pompa_mati'.encode())'''

            elif tag=='musik':
                print("buka musik ke Youtube")
                youtube_link = 'https://www.youtube.com/watch?v=yKNxeF4KMsY'  # Link VIDEO_ID
                webbrowser.open(youtube_link)

            elif tag=='terminal_hidup':
                '''arduino.write('terminal_hidup'.encode())'''
            elif tag=='terminal_mati':
                '''arduino.write('terminal_mati'.encode())'''
            
            elif tag=='pergi':
                '''arduino.write('pergi'.encode())'''
            
            elif tag=='lampusemua_hidup':
                '''arduino.write('lampusemua_hidup'.encode())'''
            elif tag=='lampusemua_mati':
                '''arduino.write('lampusemua_mati'.encode())'''
                
            elif tag=='otolampu_mati':
                '''arduino.write('otolampu_mati'.encode())'''
            elif tag=='otolampu_hidup':
                '''arduino.write('otolampu_hidup'.encode())'''

            elif tag=='otopompa_mati':
                '''arduino.write('otopompa_mati'.encode())'''
            elif tag=='otopompa_hidup':
                '''arduino.write('otopompa_hidup'.encode())'''

            elif tag=='suhu':
                '''arduino.write('suhu'.encode())
                response_from_arduino = arduino.readline().decode().strip()
                # Check if the response starts with "suhu:"
                if response_from_arduino.startswith("suhu:"):
                    try:
                        # Extract the temperature value from the response
                        _, temperature_str = response_from_arduino.split(':')
                        # Convert the temperature value to a float
                        temperature = int(temperature_str)
                        # Speak the temperature value
                        speak(f"{temperature} derajat Celsius")
                    except ValueError:
                        print("Error parsing temperature value.")'''
            
            elif tag=='jam':
                current_time = datetime.datetime.now().strftime("%H:%M")
                speak(current_time)
            elif tag=='tanggal':
                current_date = datetime.datetime.now().strftime("%d-%m-%Y")
                speak(current_date)
            elif tag=='hari':
                locale.setlocale(locale.LC_TIME, 'id_ID')
                current_day = datetime.datetime.now().strftime("%A")
                speak(current_day)

            print(return_list)
            print("-" * 50)

# Call the modified start_chat_with_voice function to start the chat with voice input
start_chat_with_voice()


