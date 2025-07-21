# import libraries
import warnings                            # Mengimpor library untuk menangani peringatan
warnings.filterwarnings("ignore")          # Mengabaikan semua peringatan agar tidak mengganggu output program
import nltk                                # Mengimpor library Natural Language Toolkit (NLTK) untuk pengolahan bahasa alami
from nltk.stem import WordNetLemmatizer    # Mengimpor kelas WordNetLemmatizer dari NLTK untuk melakukan lemmatization
import json                                # Mengimpor library JSON untuk membaca dan menulis data dalam format JSON
import pickle                              # Mengimpor library Pickle untuk serialisasi dan deserialisasi objek Python
import serial                              # Mengimpor library PySerial untuk komunikasi serial (misalnya dengan perangkat keras)
import webbrowser                          # Mengimpor library webbrowser untuk membuka URL di browser web
from tensorflow import keras               # Mengimpor library TensorFlow untuk deep learning
import numpy as np                         # Mengimpor library NumPy untuk operasi numerik dengan array
from keras.models import Sequential        # Mengimpor kelas Sequential dari Keras untuk membangun model neural network secara berurutan
from keras.layers import Dense, Activation, Dropout  # Mengimpor lapisan Dense, Activation, dan Dropout dari Keras untuk membangun neural network
from keras.optimizers import SGD           # Mengimpor optimizer Stochastic Gradient Descent (SGD) dari Keras untuk melatih model
import random                              # Mengimpor library random untuk menghasilkan angka acak
from keras.models import load_model        # Mengimpor fungsi load_model dari Keras untuk memuat model yang sudah dilatih

# Inisialisasi koneksi serial untuk Arduino pertama
arduino1 = serial.Serial('COM4', 115200)

# Inisialisasi koneksi serial untuk Arduino kedua
arduino2 = serial.Serial('COM3', 115200)

# load the saved model file
model = load_model('model.h5')                      # Memuat model neural network yang telah dilatih dari file 'model.h5'
intents = json.loads(open("intents.json").read())   # Membaca file 'intents.json' dan memuat isinya ke dalam variabel intents sebagai objek JSON
words = pickle.load(open('words.pkl','rb'))         # Memuat daftar kata yang telah diproses dari file 'words.pkl' menggunakan Pickle
classes = pickle.load(open('classes.pkl','rb'))     # Memuat daftar kelas yang telah diproses dari file 'classes.pkl' menggunakan Pickle
lemmatizer = WordNetLemmatizer()                    # Membuat instance dari WordNetLemmatizer untuk digunakan dalam lemmatization

print(classes)

####
def clean_up_sentence(sentence):                    # Mendefinisikan fungsi dengan nama 'clean_up_sentence' yang menerima satu parameter 'sentence'
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)   # Menggunakan fungsi word_tokenize dari NLTK untuk membagi kalimat menjadi kata-kata
    # stem each word - membuat bentuk singkat untuk kata
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]  # Menggunakan lemmatizer untuk mengubah setiap kata menjadi bentuk dasarnya
    return sentence_words                           # Mengembalikan array dari kata-kata yang telah di-tokenisasi dan di-lemmatize


# return bag of words array: 0 atau 1 untuk setiap kata di bag yang ada dalam kalimat
def bow(sentence, words, show_details=True):        # Mendefinisikan fungsi dengan nama 'bow' yang menerima tiga parameter: 'sentence', 'words', dan 'show_details' (default True)
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)    # Memanggil fungsi 'clean_up_sentence' untuk mengubah kalimat menjadi token kata yang telah di-lemmatize
    # bag of words 
    bag = [0]*len(words)                            # Membuat vektor 'bag' dengan panjang yang sama dengan daftar 'words', diisi dengan nilai nol
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # tetapkan 1 jika kata saat ini berada pada vocabulary
                bag[i] = 1                          # Mengisi posisi yang sesuai dalam 'bag' dengan nilai 1 jika kata ditemukan dalam daftar 'words'
                if show_details:
                    print ("found in bag: %s" % w)  # Jika 'show_details' adalah True, mencetak kata yang ditemukan dalam 'bag'
    return(np.array(bag))                           # Mengembalikan 'bag' sebagai array NumPy


def predict_class(sentence, model):                 # Mendefinisikan fungsi dengan nama 'predict_class' yang menerima dua parameter: 'sentence' dan 'model'
    # memfilter prediksi di bawah ambang batas
    p = bow(sentence, words, show_details=False)    # Memanggil fungsi 'bow' untuk mengubah kalimat menjadi representasi bag-of-words
    res = model.predict(np.array([p]))[0]           # Menggunakan model untuk memprediksi kelas berdasarkan representasi bag-of-words, kemudian mengambil hasil prediksi pertama
    error = 0.25                                    # Menetapkan nilai threshold untuk mengabaikan prediksi dengan probabilitas di bawah 0.25
    results = [[i, r] for i, r in enumerate(res) if r > error]  # Membuat daftar hasil prediksi yang probabilitasnya lebih besar dari threshold
    # urutkan berdasarkan kekuatan probabilitas
    results.sort(key=lambda x: x[1], reverse=True)  # Mengurutkan hasil prediksi berdasarkan probabilitas secara menurun
    global return_list                              # Mendefinisikan variabel global 'return_list'
    return_list = []                                # Menginisialisasi 'return_list' sebagai daftar kosong
    
    for r in results:                               # Melakukan iterasi pada setiap hasil prediksi yang lolos threshold
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})  # Menambahkan dictionary dengan intent dan probabilitas ke dalam 'return_list'
    return return_list                              # Mengembalikan 'return_list' yang berisi daftar intent dan probabilitas

#####

# fungsi untuk mendapatkan respon dari model
def getResponse(ints, intents_json):                # Mendefinisikan fungsi dengan nama 'getResponse' yang menerima dua parameter: 'ints' dan 'intents_json'
    global tag                                      # Mendeklarasikan variabel 'tag' sebagai variabel global
    tag = ints[0]['intent']                         # Mengambil intent dari hasil prediksi pertama dalam 'ints' dan menyimpannya dalam variabel 'tag'
    list_of_intents = intents_json['intents']       # Mengambil daftar intent dari objek JSON 'intents_json'
    for i in list_of_intents:                       # Melakukan iterasi pada setiap intent dalam 'list_of_intents'
        if i['tag'] == tag:                         # Mengecek apakah tag intent sesuai dengan 'tag' yang diambil dari prediksi
            result = random.choice(i['responses'])  # Memilih secara acak satu respon dari daftar respon yang sesuai dengan 'tag'
            break                                   # Keluar dari loop setelah menemukan tag yang sesuai
    return result                                   # Mengembalikan respon yang dipilih secara acak


# fungsi untuk memprediksi kelas dan mendapatkan responnya
def chatbot_response(text):                         # Mendefinisikan fungsi dengan nama 'chatbot_response' yang menerima satu parameter: 'text'
    ints = predict_class(text, model)               # Memanggil fungsi 'predict_class' untuk memprediksi kelas dari teks input menggunakan model yang diberikan, dan menyimpan hasilnya dalam 'ints'
    res = getResponse(ints, intents)                # Memanggil fungsi 'getResponse' untuk mendapatkan respon yang sesuai berdasarkan intent yang diprediksi, dan menyimpan hasilnya dalam 'res'
    return res                                      # Mengembalikan respon yang dihasilkan oleh fungsi 'getResponse'

#####

import speech_recognition as sr

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

def voice():                                        # Mendefinisikan fungsi dengan nama 'voice' tanpa parameter
    r = sr.Recognizer()                             # Membuat instance dari Recognizer untuk mengenali ucapan
    while True:                                     # Memulai loop tak terbatas untuk terus mendengarkan input suara
        try:                                        # Blok try untuk menangani potensi kesalahan
            with sr.Microphone(device_index=1) as source:  # Menggunakan mikrofon dengan device index 1 sebagai sumber input
                r.adjust_for_ambient_noise(source)  # Menyesuaikan pengenalan ucapan dengan kebisingan lingkungan
                print("Berbicara")                  # Mencetak pesan untuk memberi tahu pengguna untuk mulai berbicara
                r.pause_threshold = 1               # Menetapkan ambang batas jeda ke 1 detik untuk mengenali akhir ucapan
                audio = r.listen(source)            # Mendengarkan dan merekam input suara dari mikrofon

            print("Mengolah")                       # Mencetak pesan bahwa suara sedang diproses
            query = r.recognize_google(audio, language='id')  # Menggunakan Google Web Speech API untuk mengenali ucapan dan mengonversinya ke teks dalam bahasa Indonesia
            print("Kamu Berbicara: " + query)       # Mencetak teks hasil pengenalan ucapan
            return query                            # Mengembalikan teks hasil pengenalan ucapan
        except Exception as e:                      # Blok except untuk menangani kesalahan yang mungkin terjadi selama proses pengenalan ucapan
            print("Tidak dapat mendengar apapun")   # Mencetak pesan kesalahan jika tidak ada ucapan yang dikenali
            continue                                # Melanjutkan loop dan kembali ke blok mendengarkan ucapan

from gtts import gTTS               # Mengimpor kelas gTTS dari modul gtts untuk mengonversi teks menjadi suara
from io import BytesIO              # Mengimpor kelas BytesIO dari modul io untuk menangani file dalam memori sebagai objek byte
import pygame                       # Mengimpor modul pygame untuk memutar file audio

def speak(text):                            # Mendefinisikan fungsi dengan nama 'speak' yang menerima satu parameter: 'text'
    pygame.init()                           # Menginisialisasi semua modul pygame
    pygame.mixer.init()                     # Menginisialisasi mixer pygame untuk memutar suara
    tts_obj = gTTS(text=text, lang="id")    # Membuat objek gTTS dengan teks yang diberikan dalam bahasa Indonesia
    mp3_fp = BytesIO()                      # Membuat objek BytesIO untuk menyimpan file audio dalam memori
    tts_obj.write_to_fp(mp3_fp)             # Menulis output audio gTTS ke objek BytesIO
    mp3_fp.seek(0)                          # Mengatur posisi pointer ke awal objek BytesIO
    pygame.mixer.music.load(mp3_fp)         # Memuat file audio dari objek BytesIO ke dalam mixer pygame
    pygame.mixer.music.play()               # Memutar file audio yang dimuat
    while pygame.mixer.music.get_busy():    # Looping selama musik sedang dimainkan
        pygame.time.Clock().tick(5)         # Menunggu selama 5 milidetik sebelum memeriksa kembali status pemutaran musik

import datetime                     # Mengimpor modul datetime untuk bekerja dengan tanggal dan waktu
import locale                       # Mengimpor modul locale untuk mengatur lokalitas aplikasi


def start_chat_with_voice():                # Mendefinisikan fungsi dengan nama 'start_chat_with_voice' tanpa parameter
    print("Bot: Saya asisten mu! Katakan apa yang kamu butuhkan.\n\n")   # Mencetak pesan awal untuk pengguna
    while True:                             # Memulai loop tak terbatas untuk percakapan berkelanjutan
        inp = voice().lower()               # Menggunakan input suara dan mengonversinya ke huruf kecil
        if inp == "end":                    # Memeriksa apakah input adalah "end"
            break                           # Keluar dari loop jika input adalah "end"
        if inp == '' or inp == '*':         # Memeriksa apakah input kosong atau tanda '*'
            print('Please re-phrase your query!')   # Mencetak pesan kesalahan jika input tidak valid
            print("-" * 50)                 # Mencetak garis pemisah untuk kejelasan
        else:                               # Jika input valid
            response = chatbot_response(inp) # Mendapatkan respon chatbot berdasarkan input
            print(f"Bot: {response}" + '\n') # Mencetak respon chatbot
            speak(response)                 # Mengucapkan respon menggunakan pygame

            # Code for sending commands/kode ke Arduino
            if tag=='lampu1_hidup':                     # Kelas ini terpanggil dalam hasil pengolahan chatbot maka
                arduino1.write('lampu1_hidup'.encode()) # kode/commands berikut akan dikirim ke arduino 1
            elif tag=='lampu1_mati':
                arduino1.write('lampu1_mati'.encode())
            elif tag=='lampu2_hidup':
                arduino1.write('lampu2_hidup'.encode())
            elif tag=='lampu2_mati':
                arduino1.write('lampu2_mati'.encode())
            elif tag=='lampu3_hidup':
                arduino1.write('lampu3_hidup'.encode())
            elif tag=='lampu3_mati':
                arduino1.write('lampu3_mati'.encode())
            
            elif tag=='kipas_hidup':
                arduino1.write('kipas_hidup'.encode())
            elif tag=='kipas_naik':
                arduino1.write('kipas_naik'.encode())
            elif tag=='kipas_turun':
                arduino1.write('kipas_turun'.encode())    
            elif tag=='kipas_mati':
                arduino1.write('kipas_mati'.encode())
            
            elif tag=='ac_hidup':
                arduino1.write('ac_hidup'.encode())
            elif tag=='ac_naik':
                arduino1.write('ac_naik'.encode())
            elif tag=='ac_turun':
                arduino1.write('ac_turun'.encode())
            elif tag=='ac_mati':
                arduino1.write('ac_mati'.encode())

            elif tag=='kunci_hidup':
                arduino1.write('kunci_hidup'.encode())
            elif tag=='kunci_mati':
                arduino1.write('kunci_mati'.encode())

            elif tag=='kran_mati':
                arduino1.write('kran_mati'.encode())
            elif tag=='kran_hidup':
                arduino1.write('kran_hidup'.encode())
          
            elif tag=='tirai_hidup':
                arduino1.write('tirai_hidup'.encode())
            elif tag=='tirai_mati':
                arduino1.write('tirai_mati'.encode())

            elif tag=='pompa_hidup':
                arduino1.write('pompa_hidup'.encode())
            elif tag=='pompa_mati':
                arduino1.write('pompa_mati'.encode())

            elif tag=='musik':
                print("buka musik ke Youtube")
                youtube_link = 'https://www.youtube.com/watch?v=yKNxeF4KMsY'  # Link VIDEO_ID
                webbrowser.open(youtube_link)

            elif tag=='terminal_hidup':
                arduino1.write('terminal_hidup'.encode())
            elif tag=='terminal_mati':
                arduino1.write('terminal_mati'.encode())
            
            elif tag=='pergi':
                arduino1.write('pergi'.encode())
            
            elif tag=='lampusemua_hidup':
                arduino1.write('lampusemua_hidup'.encode())
            elif tag=='lampusemua_mati':
                arduino1.write('lampusemua_mati'.encode())
                
            elif tag=='otolampu_mati':
                arduino1.write('otolampu_mati'.encode())
            elif tag=='otolampu_hidup':
                arduino1.write('otolampu_hidup'.encode())

            elif tag=='otopompa_mati':
                arduino1.write('otopompa_mati'.encode())
            elif tag=='otopompa_hidup':
                arduino1.write('otopompa_hidup'.encode())

            elif tag=='suhu':
                arduino2.write('suhu'.encode())    # baca hasil sensor suhu dari arduino 2
                response_from_arduino = arduino2.readline().decode().strip()
                # Periksa apakah respons dimulai dengan "suhu:"
                if response_from_arduino.startswith("suhu:"):
                    try:
                        # Ekstrak nilai suhu dari respons
                        _, temperature_str = response_from_arduino.split(':')
                        # Ubah nilai suhu menjadi bilangan bulat
                        temperature = int(temperature_str)
                        # Ucapkan nilai suhu
                        speak(f"{temperature} derajat Celsius")
                    except ValueError:
                        print("Error parsing temperature value.")
            
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

# Panggil fungsi start_chat_with_voice yang dimodifikasi untuk memulai obrolan dengan input suara
start_chat_with_voice()


