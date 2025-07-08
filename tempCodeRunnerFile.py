from flask import Flask, request, render_template, jsonify
import os
import librosa
import speech_recognition as sr
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import unicodedata

app = Flask(__name__)

# Load saved model and tools
model = tf.keras.models.load_model('model/best_model.h5')

with open('model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
    print("✅ Flask tokenizer vocab size:", len(tokenizer.word_index)) #
    print("🔢 Flask test sequence:", tokenizer.texts_to_sequences(["তোকে মেরে ফেলবো"]))#


with open('model/label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

MAX_SEQUENCE_LENGTH = 100
BENGALI_STOPWORDS = set([
    "আমি", "আমরা", "তুমি", "তোমরা", "সে", "তাহারা", "এই", "সেই",
    "এখানে", "সেখানে", "কি", "কেন", "কখন", "কিভাবে", "হয়", "নয়",
    "থেকে", "করে", "সব", "কিছু"
])

def preprocess_bengali_text(text):
    text = text.lower()
    text = re.sub(r'[^\u0980-\u09FF\s]', '', text)
    text = ' '.join(text.split())
    text = unicodedata.normalize('NFKC', text)
    tokens = re.findall(r'[\u0980-\u09FF]+', text)
    filtered_tokens = [word for word in tokens if word not in BENGALI_STOPWORDS]
    return ' '.join(filtered_tokens)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    audio_file = request.files['audio']
    file_path = 'temp.wav'
    audio_file.save(file_path)

# 🔁 Convert to proper PCM WAV
#import librosa
    #import soundfile as sf

    #y, sr_rate = librosa.load(file_path, sr=None)
    #sf.write(file_path, y, sr_rate, subtype='PCM_16')
    from pydub import AudioSegment

    try:
        audio = AudioSegment.from_file(file_path)
        audio.export(file_path, format="wav", codec="pcm_s16le")
    except Exception as e:
        print("⚠️ Audio conversion failed:", e)

# Transcribe
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:

        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio, language="bn-IN")
            print(f"Transcribed Text: {text}")
        except:
            return jsonify({'result': 'Could not transcribe'})

    preprocessed = preprocess_bengali_text(text)
    print("🧹 Preprocessed Text:", preprocessed)

    seq = tokenizer.texts_to_sequences([preprocessed])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    prob = model.predict(padded)[0][0]

    # Apply prediction with a stricter threshold
    label = (prob > 0.9).astype('int32')

# Keyword-based override for playful or indirect tones
    override_keywords = [
        'ট্রিট', 'ট্রিটটা', 'ঘুমিয়ে', 'হাসছি', 'শেষ', 'মজা', 'চলে যাও', 'চুপ', 'ভালো',
        'চাইনা', 'চাই', 'তুমি কেমন', 'সময় আসলে', 'ভেবে', 'নিজে থেকে', 'দোষ', 
        'না বলবো', 'না দেখে', 'চাপ', 'বাড়াবাড়ি', 'আগে', 'তোর ক্ষতি', 'চুপ করে', 
        'তোর কিন্তু', 'আমি চাই', 'নিজের ভালো', 'না খাওয়ালে', 'খারাপ হবে', 'নিজে', 
        'নিজের দোষ', 'হয়ে যাবে', 'খুব বেশি', 'শুধু বলছি', 'রাগ', 'রাগ করেছি', 'রাগ'
        'রাগ করছি', 'রেগে', 'রেগে আছি', 'শেষ হয়ে গেছে', 'সবকিছু শেষ', 'সব শেষ',
        'আমি শেষ মানুষ', 'ছেড়ে দেবো'
    ]

    if label == 1:
        for word in override_keywords:
            if word in preprocessed.split() and prob < 0.995:
                print(f"🔁 Soft override: matched keyword '{word}', prob={prob:.3f}")
                label = 0
                break


    # Step 3: Hard override (false negatives → Force Threat)
    dangerous_keywords = [
        'মারে', 'মেরে', 'মারবো', 'মারবো', 'মার দেবো',
        'তোমাকে মারবো', 'তোকে মারবো', 'খাবি', 'খাবে', 'খাবো',
        'ফেলবো', 'শেষ করে', 'শেষ করবো', 'ঝুলিয়ে', 'খুন', 'ধরবো',
        'বঁটি', 'গায়ে হাত', 'পিটিয়ে', 'গলা টিপে', 'খতম', 'ছুরি',
        'গুলি', 'গুলি করবো', 'গুলি করবে', 'হত্যা'
    ]

    if label == 0 and prob < 0.3:
        for word in dangerous_keywords:
            if word in preprocessed.split():
                print(f"⚠️ Hard override to Threat: matched '{word}', prob={prob:.3f}")
                label = 1
                break

    result = "Threat" if label == 1 else "No Threat"

    #label = (prob > 0.5).astype('int32')
    #result = label_encoder.inverse_transform([label])[0]
    #result = "Threat" if label == 1 else "No Threat"
    print("📈 Flask Raw Probability:", prob)#
    print("🔮 Flask Predicted Label:", result) #


    #return jsonify({'result': str(result)})
    return jsonify({'result': result})


if __name__ == '__main__':
    app.run(debug=True)
