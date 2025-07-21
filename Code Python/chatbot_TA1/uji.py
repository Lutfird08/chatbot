# import necessary libraries
import warnings
warnings.filterwarnings("ignore")
import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import serial

from tensorflow import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
from keras.models import load_model

# Inisialisasi koneksi serial dengan Arduino
arduino = serial.Serial('COM6', 115200)

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

def start_chat():
    print("Bot: Saya asisten mu! Katakan apa yang kamu butuhkan.\n\n")
    while True:
        inp = str(input()).lower()
        if inp.lower()=="end":
            break
        if inp.lower()== '' or inp.lower()== '*':
            print('Please re-phrase your query!')
            print("-"*50)

        else:
            print(f"Bot: {chatbot_response(inp)}"+'\n')
            if tag=='ac_hidup':
                print("ASOOY")
                arduino.write('ac_hidup'.encode())
            elif tag=='ac_mati':
                print("ASOOY")
                arduino.write('ac_mati'.encode())
            elif tag=='ac_naik':
                arduino.write('ac_naik'.encode())
            elif tag=='ac_turun':
                arduino.write('ac_turun'.encode())
            elif tag=='kunci_hidup':
                arduino.write('kunci_hidup'.encode())
            elif tag=='kunci_mati':
                arduino.write('kunci_mati'.encode())
            elif tag=='kran_hidup':
                arduino.write('kran_hidup'.encode())
            elif tag=='kran_mati':
                arduino.write('kran_mati'.encode())
            elif tag=='lampu1_hidup':
                arduino.write('lampu1_hidup'.encode())
            elif tag=='lampu1_mati':
                arduino.write('lampu1_mati'.encode())
            elif tag=='lampu2_hidup':
                arduino.write('lampu2_hidup'.encode())
            elif tag=='lampu2_mati':
                arduino.write('lampu2_mati'.encode())
            elif tag=='lampu3_hidup':
                arduino.write('lampu3_hidup'.encode())
            elif tag=='lampu3_mati':
                arduino.write('lampu3_mati'.encode())
            
            elif tag=='kipas_hidup':
                arduino.write('kipas_hidup'.encode())
            elif tag=='kipas_mati':
                arduino.write('kipas_mati'.encode())
            

            print(return_list)
            print("-"*50)

#####



# start the chat bot
start_chat()
