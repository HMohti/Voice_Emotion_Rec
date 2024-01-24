import os
import glob
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2

class VoiceRecognizer:
    def __init__(self, pos_path, neg_path):
        self.pos_path = pos_path
        self.neg_path = neg_path
        self.model = self._build_model()
    
    def _build_model(self):
        model = Sequential([
            Conv2D(16, (3,3), activation='relu', input_shape=(14991, 257, 1)),
            # we can add regul L2 and dropout to improve the val precision if it's low
            # Conv2D(16, (3,3), activation='relu', input_shape=(14991, 257, 1), kernel_regularizer=l2(0.0005)), # rajout regul l2
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(16, (3,3), activation='relu'),
            # Conv2D(16, (3,3), activation='relu', kernel_regularizer=l2(0.0005)),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            # Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
            # Dropout(0.3),  # rajout du dropout 30%
            Dense(1, activation='sigmoid')
        ])
        model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])
        return model

    def load_wav_16k_mono(self, filename):
        file_contents = tf.io.read_file(filename)
        wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
        wav = tf.squeeze(wav, axis=-1)
        sample_rate = tf.cast(sample_rate, dtype=tf.int64)
        wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
        return wav

    def preprocess(self, file_path, label):
        wav = self.load_wav_16k_mono(file_path)
        wav = wav[:480000]
        zero_padding = tf.zeros([480000] - tf.shape(wav), dtype=tf.float32)
        wav = tf.concat([zero_padding, wav], 0)
        spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
        spectrogram = tf.abs(spectrogram)
        spectrogram = tf.expand_dims(spectrogram, axis=2)
        return spectrogram, label

    def load_data(self):
        neg_files = glob.glob(self.neg_path + '/**/*.wav', recursive=True)
        pos_files = glob.glob(self.pos_path + '/*.wav')

        # Imprimer le nombre de fichiers dans chaque répertoire
        print(f"Number of files in POS: {len(pos_files)}")
        print(f"Number of files in NEG: {len(neg_files)}")

        pos = tf.data.Dataset.from_tensor_slices(pos_files)
        neg = tf.data.Dataset.from_tensor_slices(neg_files)

        positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos_files)))))
        negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg_files)))))

        data = positives.concatenate(negatives)
        data = data.map(self.preprocess)
        data = data.cache()
        data = data.shuffle(buffer_size=100)
        data = data.batch(16)
        data = data.prefetch(8)

        return data
    
    def split_data(self, data, train_size):
        train_data = data.take(train_size)
        test_data = data.skip(train_size)
        return train_data, test_data

    def calculate_audio_lengths(self, directory):
        lengths = []
        file_names = []
        for file in os.listdir(directory):
            if not file.endswith('.wav'):
                continue
            full_path = os.path.join(directory, file)
            file_names.append(file)
            tensor_wave = self.load_wav_16k_mono(full_path)
            lengths.append(len(tensor_wave))

        print("Processed audio files:", file_names)
        return lengths

    def train_and_save_model(self, train_data, test_data, model_file_name):
        hist = self.model.fit(train_data, epochs=4, validation_data=test_data)
        self.model.save(model_file_name)
        return hist

    def predict_and_evaluate(self, test_data):
        X_test, y_test = test_data.as_numpy_iterator().next()
        yhat = self.model.predict(X_test)
        yhat_binary = [1 if prediction > 0.5 else 0 for prediction in yhat]

        # Convertir yhat_binary en tenseur de type int32
        yhat_binary_tensor = tf.cast(yhat_binary, tf.int32)

        # Convertir y_test en tenseur de type int32
        y_test_tensor = tf.cast(y_test, tf.int32)

        print("X_test shape:", X_test.shape)
        print("Predictions:", yhat)
        print("Binary Predictions:", yhat_binary_tensor)
        print("Actual:", y_test_tensor)

        print("Sum of Predictions:", tf.math.reduce_sum(yhat_binary_tensor))
        print("Sum of Actual:", tf.math.reduce_sum(y_test_tensor))
        print("Actual Labels (int):", y_test_tensor)


pos_path = 'newdata/hanae'
neg_path = 'newdata/autres'

vr = VoiceRecognizer(pos_path, neg_path)
data = vr.load_data()
train_data, test_data = vr.split_data(data, 3)

lengths = vr.calculate_audio_lengths(pos_path)
print("Audio lengths:", lengths)
print("Mean length:", tf.math.reduce_mean(lengths))
print("Min length:", tf.math.reduce_min(lengths))
print("Max length:", tf.math.reduce_max(lengths))

hist = vr.train_and_save_model(train_data, test_data, 'hanae_voice_recognition_model_V3.h5')
print(hist)

vr.predict_and_evaluate(test_data)
