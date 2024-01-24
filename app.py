from flask import Flask, request, jsonify
import numpy as np 
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras.models import load_model
from pydub import AudioSegment
from itertools import groupby
import librosa
from flask_cors import CORS
from flask_socketio import SocketIO
import socketio as sio_client


# Load the models
voice_model = load_model('hanae_voice_recognition_model_V3.h5')
emotion_model = load_model('Speech_Emotion_Recognition_Model_6.h5')

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app)
sio = sio_client.Client()
sio.connect('http://localhost:4001')

def get_audio_features(audio_path, sampling_rate=20000):
    X, _ = librosa.load(audio_path, res_type='kaiser_fast', duration=2.5, sr=sampling_rate*2, offset=0.5)
    y_harmonic, y_percussive = librosa.effects.hpss(X)
    pitches, magnitudes = librosa.piptrack(y=X, sr=sampling_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sampling_rate, n_mfcc=20), axis=1)
    pitches = np.trim_zeros(np.mean(pitches, axis=1))[:20]
    magnitudes = np.trim_zeros(np.mean(magnitudes, axis=1))[:20]
    C = np.mean(librosa.feature.chroma_cqt(y=y_harmonic, sr=sampling_rate), axis=1)
    feature_vector = np.hstack([mfccs, pitches, magnitudes, C])
    return feature_vector

def preprocess_audio_for_emotion(audio_path):
    features = get_audio_features(audio_path)
    features = np.expand_dims(features, axis=0)
    features = np.expand_dims(features, axis=2)
    return features

@app.route('/process_audio', methods=['POST'])
def process_audio():
    audio_file = request.files['audio_data']
    audio_file.save("audio_received.wav")
    audio = AudioSegment.from_wav("audio_received.wav")
    audio.export("audio_received.mp3", format="mp3")

    def load_mp3_16k_mono(filename):
        # """ Load a WAV file, convert it to a float tensor, resample to 16 kHz single-channel audio. """
        res = tfio.audio.AudioIOTensor(filename)
        # Convert to tensor and combine channels
        tensor = res.to_tensor()
        tensor = tf.math.reduce_sum(tensor, axis=1) / 2
        # Extract sample rate and cast
        sample_rate = res.rate
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        # Resample to 16 kHz
        wav = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=16000)
        return wav

    def preprocess_mp3(sample, index):
        sample = sample[0]
        zero_padding = tf.zeros([480000] - tf.shape(sample), dtype=tf.float32)
        wav = tf.concat([zero_padding, sample],0)
        spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.expand_dims(spectrogram, axis=2)
        return spectrogram

    # Voice Recognition
    query_audio = 'audio_received.mp3'
    resultat = load_mp3_16k_mono(query_audio)
    audio_slices = tf.keras.utils.timeseries_dataset_from_array(resultat, resultat, sequence_length=48000, sequence_stride=48000, batch_size=1)
    audio_slices = audio_slices.map(preprocess_mp3)
    audio_slices = audio_slices.batch(64)

    raw_predictions = voice_model.predict(audio_slices)
    print("Raw Predictions:", raw_predictions)   # This will print the raw predictions

    yhat = [1 if prediction > 0.99 else 0 for prediction in raw_predictions]
    print("YHAT:", yhat)
    # proportion_ones = sum(yhat) / len(yhat)
    # class_name = "Hanae's Voice" if proportion_ones > 0.5 else "Unknown Voice"

    postprocessed = tf.math.reduce_sum([key for key, group in groupby(yhat)]).numpy()
    class_name = "Hanae's Voice" if postprocessed >= 1 else "Unknown Voice"

    # Emotion Recognition
    audio_features_emotion = preprocess_audio_for_emotion("audio_received.wav")
    emotion_preds = emotion_model.predict(audio_features_emotion)
    emotion_index = np.argmax(emotion_preds)
    emotions = ["surprise", "sad", "neutral", "happy", "fear", "disgust", "anger"]
    predicted_emotion = emotions[emotion_index]

    print("Emitting voice recognition result:", class_name)
    print("Emitting voice recognition result:", predicted_emotion)
    sio.emit('voice_recognition_result', {"voice": class_name, "emotion": predicted_emotion})

    return jsonify({"voice": class_name, "emotion": predicted_emotion})

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(port=5001)  # Run the server on port 5001
