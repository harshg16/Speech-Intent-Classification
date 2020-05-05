import numpy as np
import pandas as pd
import re
import os
os.environ['KERAS_BACKEND']='theano' # Why theano why not
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
plt.switch_backend('agg')
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import os
import requests
import zipfile
import io

def main(zip_file_url):
    transcripts = []
    
    # zip_file_url = "https://ai-hackathon-upload.s3.ap-south-1.amazonaws.com/public/data.zip"
    r = requests.get(zip_file_url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall()
    
    for filea in z.filelist:
        demo = AudioSegment.from_wav(filea.filename)
        text = transcribe(demo)
        transcripts.append((filea.filename, text))
    return transcripts


def transcribe(demo):
    with open("C:/Users/harsh/Documents/MPSTME/AI Hackathon/Docs/AIHackathon-c3520d4807f1.json") as f:
        GOOGLE_CLOUD_SPEECH_CREDENTIALS = f.read()
    chunks = split_on_silence(demo,
                              # must be silent for at least 0.5 seconds
                              # or 500 ms. adjust this value based on user
                              # requirement. if the speaker stays silent for
                              # longer, increase this value. else, decrease it.
                              min_silence_len=1200,
    
                              # consider it silent if quieter than -16 dBFS
                              # adjust this per requirement
                              silence_thresh=-50
                              )
    text = ""
    # create a directory to store the audio chunks.
    try:
        os.mkdir('audio_chunks')
    except(FileExistsError):
        pass
    
    # move into the directory to
    # store the audio files.
    os.chdir('audio_chunks')
    
    # Create 0.5 seconds silence chunk
    chunk_silent = AudioSegment.silent(duration=10)
    
    i = 0
    # process each chunk
    for chunk in chunks:
        if chunk.duration_seconds > 59:
            for i in range(chunk.duration_seconds % 59):
                subchunk = chunk[i * 60 * 1000:((i + 1) * 60 * 1000) - 1000]
                subchunk = chunk_silent + subchunk + chunk_silent
                # specify the bitrate to be 192 k
                subchunk.export("./chunk{0}.wav".format(i), bitrate='192k', format="wav")
                
                # the name of the newly created chunk
                filename = 'chunk' + str(i) + '.wav'
                
                
                # get the name of the newly created chunk
                # in the AUDIO_FILE variable for later use.
                file = filename
                
                # create a speech recognition object
                r = sr.Recognizer()
                
                # recognize the chunk
                with sr.AudioFile(file) as source:
                    audio = r.record(source)
                try:
                    # try converting it to text
                    text1 = r.recognize_google_cloud(audio, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS, language='en-IN')
                    # write the output to the file.
                    text += text1
                
                # catch any errors.
                except sr.UnknownValueError:
                
                except sr.RequestError as e:
                
                i += 1
        else:
            # add 0.5 sec silence to beginning and
            # end of audio chunk. This is done so that
            # it doesn't seem abruptly sliced.
            chunk = chunk_silent + chunk + chunk_silent
            
            # export audio chunk and save it in
            # the current directory.
            # specify the bitrate to be 192 k
            chunk.export("./chunk{0}.wav".format(i), bitrate='192k', format="wav")
            
            # the name of the newly created chunk
            filename = 'chunk' + str(i) + '.wav'
            
            
            # get the name of the newly created chunk
            # in the AUDIO_FILE variable for later use.
            file = filename
            
            # create a speech recognition object
            r = sr.Recognizer()
            
            # recognize the chunk
            with sr.AudioFile(file) as source:
                audio = r.record(source)
            try:
                # try converting it to text
                text1 = r.recognize_google_cloud(audio, credentials_json=GOOGLE_CLOUD_SPEECH_CREDENTIALS, language='en-IN')
                # write the output to the file.
                text += text1
            
            # catch any errors.
            except sr.UnknownValueError:
            
            except sr.RequestError as e:
            
            i += 1
    
    os.chdir('..')
    return text


zip_file_url = "https://ai-hackathon-upload.s3.ap-south-1.amazonaws.com/public/data.zip"
t = main(zip_file_url)

##CLASSIFIER

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

dff = pd.read_csv(r'C:\Users\harsh\Documents\MPSTME\AI Hackathon\Final_Data.csv',engine='python')
dff = dff.dropna()
dff = dff.reset_index(drop=True)

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')

def clean_text(text):
    #ALL CHARACTERS ARE CONVERTED INTO LOWER CASE
    text = text.lower()
    #SEPARATING CHARACTERS ARE REMOVED
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    #INDIVIDUAL NUMBERS AND OPERATORS ARE REMOVED
    text = BAD_SYMBOLS_RE.sub('', text)
    return text

dff['sentence'] = dff['sentence'].apply(clean_text)

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(dff["sentence"])
sequences = tokenizer.texts_to_sequences(dff["sentence"])
word_index = tokenizer.word_index

macronum=sorted(set(dff['label']))
macro_to_id = dict((note, number) for number, note in enumerate(macronum))

def fun(i):
    return macro_to_id[i]

dff['label']=dff['label'].apply(fun)

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(dff["label"]))

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, random_state = 42, stratify = labels)

embeddings_index = {}
f = open(r'C:\Users\harsh\Documents\MPSTME\AI Hackathon\glove.6B.100d.txt',encoding='utf8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,trainable=True)

from keras.models import load_model
model = load_model('C:/Users/harsh/Documents/MPSTME/AI Hackathon/model_cnn.hdf5')

def predict_class(inp):
    inp = tokenizer.texts_to_sequences([inp])
    inp = pad_sequences(inp, maxlen=MAX_SEQUENCE_LENGTH)
    p = np.argmax(model.predict_on_batch(inp),axis = 1)[0]
    return macronum[p]

def transform_text(text):
    text = ' '.join(word for word in text.split() if word in bank)
    return text

bdf = pd.read_csv("C:/Users/harsh/Documents/MPSTME/AI Hackathon/Docs/bank.csv")
bank = []
ans = []
for x in bdf.values:
    bank.append(x[0])
for x in t:
    ans.append((predict_class((transform_text(x[1])))))