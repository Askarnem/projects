from spotipy.exceptions import SpotifyException
import tensorflow as tf
import transformers
import numpy as np
import pandas as pd
import sklearn
import csv
import nltk
from nltk.corpus import stopwords
import random
import torch
import re
import seaborn as sns
import torch
from torch import nn, optim
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import warnings
import json
from speech_recognition import Microphone, Recognizer, UnknownValueError
import spotipy as sp
from spotipy.oauth2 import SpotifyOAuth
from spotipy import Spotify
import requests
import os
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from collections import Counter
import math
from bs4 import BeautifulSoup
from IPython.display import display
import gtts
from playsound import playsound
import PySimpleGUI as sg
import webbrowser

warnings.filterwarnings("ignore")

# Create used interface using PySimpleGUI
layout = [[sg.Text("Emily smart assistant")],[sg.Button("Speak")], [sg.Output(size=(88, 20), font='Courier 10')]]
window = sg.Window("Emily", layout)

# Define microphone and recognizer
r = Recognizer()
m = Microphone()



# Load ALBERT model
from transformers import AlbertForSequenceClassification
model = AlbertForSequenceClassification.from_pretrained('albert-base-v1', num_labels = 3)
model.load_state_dict(torch.load('/Users/askarnemaev/Desktop/Project final/AlbertModelparams.pt',  map_location=torch.device('cpu')))
#model = model.to(device)

# Import tokenizer and set the Max Length after tokenization
from transformers import AlbertTokenizer
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v1')
MAX_LEN=40

while True:
    event, values = window.read()
    
    if event == "Speak":
    
        # Record audio
        print('Listening')
        with m as source:
            r.adjust_for_ambient_noise(source=source)
            audio = r.listen(source=source)

        #Recognize audio
        command = 'None'
        try:
            command = r.recognize_google(audio_data=audio).lower()
        except UnknownValueError:
            print('no audio recognized')

        print(command)

        # If there's a GPU available...
        if torch.cuda.is_available():    

            # Tell PyTorch to use the GPU.    
            device = torch.device("cuda")
            print('There are %d GPU(s) available.' % torch.cuda.device_count())
            print('We will use the GPU:', torch.cuda.get_device_name(0))
        # If not...
        else:
            print('No GPU available, using the CPU instead.')
            device = torch.device("cpu")


        #Create function to formulate prediction on test set
        def predict_sentiment(text):
            review_text = text

            encoded_review = tokenizer.encode_plus(
            review_text,
            max_length=MAX_LEN,
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=False,
            return_attention_mask=True,
            return_tensors='pt',
            )

            input_ids = pad_sequences(encoded_review['input_ids'], maxlen=MAX_LEN, dtype=torch.Tensor ,truncating="post",padding="post")
            input_ids = input_ids.astype(dtype = 'int64')
            input_ids = torch.tensor(input_ids) 

            attention_mask = pad_sequences(encoded_review['attention_mask'], maxlen=MAX_LEN, dtype=torch.Tensor ,truncating="post",padding="post")
            attention_mask = attention_mask.astype(dtype = 'int64')
            attention_mask = torch.tensor(attention_mask) 

            input_ids = input_ids.reshape(1,MAX_LEN).to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            outputs = outputs[0][0].cpu().detach()

            probs = F.softmax(outputs, dim=-1).cpu().detach().numpy().tolist()
            _, prediction = torch.max(outputs, dim =-1)

            #print("Positive score:", probs[1])
            #print("Negative score:", probs[0])
            #print(f'Review text: {review_text}')
            return class_names[prediction]



        class_names=['spotify','netflix','other']
        # command="I want to watch some horror movie"
        predict_com = predict_sentiment(command)
        # print(predict_com)

        ################################################################
        #SPOTIFY

        if predict_com == 'spotify':
            
            # Import required API keys and settings
            setup = pd.read_csv('/Users/askarnemaev/Desktop/Masters/NLP/project/project compile/setup.txt',
            sep='=', index_col=0, squeeze=True, header=None)
            client_id = setup['client_id']
            client_secret = setup['client_secret']
            device_name = setup['device_name']
            redirect_uri = setup['redirect_uri']
            scope = setup['scope']
            username = setup['username']


            class InvalidSearchError(Exception):
                pass
            def get_track_uri(spotify: Spotify, name: str) -> str:

                # Replace all spaces in name with '+'
                original = name
                name = name.replace(' ', '+')

                # Search for music track and return the track URI
                results = spotify.search(q=name, limit=1, type='track')
                if not results['tracks']['items']:
                    raise InvalidSearchError(f'No music named "{original}"')
                track_uri = results['tracks']['items'][0]['uri']
                return track_uri

            # Play the music
            def play_track(spotify=None, device_id=None, uri=None):
                spotify.start_playback(device_id=device_id, uris=[uri])



            # Connect to the Spotify account
            auth_manager = SpotifyOAuth(
                client_id=client_id,
                client_secret=client_secret,
                redirect_uri=redirect_uri,
                scope=scope,
                username=username)
            spotify = sp.Spotify(auth_manager=auth_manager)

            # Define device
            devices = spotify.devices()
            deviceID = None
            for d in devices['devices']:
                d['name'] = d['name'].replace('’', '\'')
                if d['name'] == device_name:
                    deviceID = d['id']
                    break

            # Pre-process the request
            words = command.lower().split()
            stop_words = ['play', 'song', 'artist', 'playlist', 'some', 'by', 'any', 'from', 'listen']
            words_clean = [i for i in words if i not in stop_words]
            name = ' '.join(words_clean)

            # Search for music track and return the track URI
            attempt = False
            try:
                uri = get_track_uri(spotify=spotify, name=name)
                play_track(spotify=spotify, device_id=deviceID, uri=uri)
                attempt = True
            except InvalidSearchError:
                print('No music found')
            except SpotifyException:
                print('')

            # Play the pre-recorded sound identifying that music is about to play in spotify app
            if attempt:
                playsound("spotify.mp3")

        ################################################################
        #NETFLIX

        elif predict_com == 'netflix':


            print('searching on Netflix')
            ps = PorterStemmer()

            # Fuctions for request pre-processing
            def remove_html(strig):
                model = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});'")
                cleaned = re.sub(model, '', strig)
                return cleaned
            def strip_html(text):
                soup = BeautifulSoup(text, "html.parser")
                return soup.get_text()
            def remove_between_square_brackets(text):
                return re.sub('\[[^]]*\]', '', text)
            def remove_special_char(string):
                return "".join(k for k in string if (k.isalnum() or k in [' ',',']))
            def denoise_text(text):
                text = remove_html(text)
                text = strip_html(text)
                text = remove_between_square_brackets(text)
                text = remove_special_char(text)
                return text.lower()
            WORD = re.compile(r"\w+")
            dic = {'movi':'','tv':'',"&":',','film':'','play':'','show':'',"'":""}
            def replace_all(text, dic):
                for i, j in dic.items():
                    text = text.replace(i, j)
                return text
            
            # Function for cosine similarity between two vectors
            def get_cosine(vec1, vec2):
                intersection = set(vec1.keys()) & set(vec2.keys())
                numerator = sum([vec1[x] * vec2[x] for x in intersection])

                sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
                sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
                denominator = math.sqrt(sum1) * math.sqrt(sum2)
                if not denominator:
                    return 0.0
                else:
                    return float(numerator) / denominator
            def text_to_vector(text):
                words = WORD.findall(text)
                return Counter(words)
            def weighted_score(row,command):
                return row.apply(lambda x: get_cosine(x,command)).to_list()
            def remove_stopwords(text):
                text_tokens = word_tokenize(text)
                tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
                return tokens_without_sw
            
            # Function for movie link request from API
            def get_film_link(
                            film_title,
                            url = "https://unogsng.p.rapidapi.com/search",
                            headers = {
                                        'x-rapidapi-key': "6a0234e36fmshaae0f31dd0bc446p12d0edjsn8c982118ef9",
                                        'x-rapidapi-host': "unogsng.p.rapidapi.com"
                                        }
                            ):
                querystring = {'query': film_title }
                
                response = requests.request("GET", url, headers=headers, params=querystring)
                
                k = json.loads(response.text)
                x = []
                LINK = 'https://www.netflix.com/sg/title/'
                for i in k['results']:
                    link = LINK  + str(i['nfid'])
                    title = i['title'].lower()
                    if title == film_title.lower():
                        r = requests.get(link).status_code
                        if r !=404:
                            x.append([i['title'],link])
                        # if x == []:
                        #     print('no movies found available in sg')
                return pd.DataFrame(x,columns=['Recommendations','Link'])

            # Function for scoring
            def get_score_vals(x,command_vec,command_list,command_list_title, weight = np.array([1,1,2,1.5,1,1])):
                l = x.values
                grade = [get_cosine(command_vec,k) for k in l[:2]] + [len(command_list & set(k)) for k in l[2:4]] + [len(command_list_title & set(k)) for k in l[4:6]]
                grade = np.array(grade) * weight
                return grade.mean()
            
            # Function to get link
            def netflix_get_link(command,vals=pd.read_pickle('/Users/askarnemaev/Desktop/Masters/NLP/project/project compile/netflix_dat_v2'), weight = np.array([1,1,2,1.5,1,1])):
                import webbrowser
                command = " ".join(remove_stopwords(replace_all(command,dic)))
                stemmed_command = " ".join([ps.stem(x) for x in word_tokenize(command)])
                command_list = set(word_tokenize(stemmed_command))
                command_vec = text_to_vector(command)
                command_list_title = set(remove_stopwords(command))
                chrome_path = 'open -a /Applications/Google\ Chrome.app %s'

                # Film search in database
                score = vals.apply(lambda k: get_score_vals(k,command_vec,command_list,command_list_title),axis=1)
                idx = (score.sort_values(ascending=False)[:5].index)
                film_list = vals.iloc[idx]['Title'].values
                y = pd.DataFrame()
                for i in film_list:
                    try:
                        x = get_film_link(i)
                        y = pd.concat([y,x])
                    except:
                        pass
                y.reset_index(inplace=True,drop=True)
                if len(y)==0:
                    return "No available movies link found"
                else:
                    recoms = y.drop_duplicates().set_index('Recommendations').rename(columns={"Recommendations":"","Link":""})
                    # Open links in browser to watch movie
                    for i in recoms[recoms.columns[0]]:
                        webbrowser.get(chrome_path).open(i)
                    playsound("netflix.mp3")

            recoms = netflix_get_link(command=command)


        else:
            # If request is not Spotify or Netflix, the output is defined below. In this part, new appliances can be added, but ALBERT has to be retrained.
            print('Appliance is not connected yet')
            playsound("other.mp3")

    elif event == sg.WIN_CLOSED:
        break