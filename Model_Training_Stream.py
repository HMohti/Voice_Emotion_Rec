import streamlit as st
import librosa
import soundfile as sf
import numpy as np
import tensorflow as tf
from pydub import AudioSegment
import os
import zipfile
from io import BytesIO
import tensorflow_io as tfio
import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
import shutil

def change_speed(waveform, speed_factor):
    return librosa.effects.time_stretch(waveform, rate=speed_factor)

def generate_noise(waveform, noise_level):
    noise = np.random.normal(0, noise_level, waveform.shape)
    return waveform + noise
 
def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

# def preprocess(segment, sample_rate=16000, duration=30):
#     # Assurez-vous que tous les fichiers audio ont la même longueur
#     segment = segment[:sample_rate * duration]
#     zero_padding = tf.zeros([sample_rate * duration] - tf.shape(segment), dtype=tf.float32)
#     segment = tf.concat([zero_padding, segment], 0)
#     spectrogram = tf.signal.stft(segment, frame_length=320, frame_step=32)
#     spectrogram = tf.abs(spectrogram)
#     spectrogram = tf.expand_dims(spectrogram, axis=2)
#     return spectrogram

def clear_folder(folder_path):
    """ Remove all files in the specified folder. """
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                st.write(f'Deleted: {file_path}')  # Logging the deletion
            except Exception as e:
                st.error(f'Failed to delete {file_path}. Reason: {e}')
    else:
        st.write(f'Folder {folder_path} does not exist. No need to clear.')

def segment_audio(file_path, segment_duration_sec, target_folder):
    # 1. Load the audio file into an AudioSegment object
    audio = AudioSegment.from_file(file_path)

    # 2. Duration of each segment in milliseconds (e.g., 30 seconds = 30*1000 milliseconds)
    length_segment = segment_duration_sec * 1000

    # 3. Create the target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)

    # 4. Cut the audio file into segments and save them in the target folder
    for i in range(0, len(audio), length_segment):
        segment = audio[i:i + length_segment]

        # Check if the duration of the segment is at least the specified duration
        if len(segment) >= length_segment:
            segment_path = os.path.join(target_folder, f"segment_{i//length_segment}.wav")
            segment.export(segment_path, format="wav")

    # 5. Optionally, create a ZIP file containing all segments
    zip_path = os.path.join(target_folder, 'segments.zip')
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, _, files in os.walk(target_folder):
            for file in files:
                if file.endswith('.wav'):
                    zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), target_folder))

    return zip_path  # Return the path to the ZIP file

def main():
    st.title("Bienvenue dans votre espace de prétraitement et d'entraînement des modèles")

    change_speed_option = st.checkbox('Modifier la Vitesse', key='change_speed')
    add_noise_option = st.checkbox('Ajouter du Bruit', key='add_noise')
    segmentation_option = st.checkbox('Segmentation', key='segmentation')
    training_option = st.checkbox('Entraînement du Modèle', key='training')

    existing_folder = 'C:\\Users\\hanae\\Documents\\VOICE_REC_ML_VX\\newdata'
    audios_folder = os.path.join(existing_folder, 'extracted_audios')

    if change_speed_option:
        speed_factor = st.slider('Facteur de Vitesse', 0.5, 2.0, 1.0, key='speed_slider')
    
    if add_noise_option:
        noise_level = st.slider('Niveau de Bruit', 0.0, 1.0, 0.1, key='noise_slider')

    uploaded_file = st.file_uploader("Téléchargez un fichier audio", type=["wav"])

    if uploaded_file is not None:
         # Save the uploaded file to a temporary path
        temp_file_path = "temp_uploaded_file.wav"
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        waveform, sample_rate = librosa.load(uploaded_file, sr=None, mono=True)

        if change_speed_option and st.button('Tester Vitesse'):
            speeded_waveform = change_speed(waveform, speed_factor)
            speeded_waveform = speeded_waveform / np.max(np.abs(speeded_waveform))
            result_path_speed = 'result_speed.wav'
            sf.write(result_path_speed, speeded_waveform, sample_rate)
            st.audio(result_path_speed)

        if add_noise_option and st.button('Tester Bruit'):
            noised_waveform = generate_noise(waveform, noise_level)
            noised_waveform = noised_waveform / np.max(np.abs(noised_waveform))
            result_path_noise = 'result_noise.wav'
            sf.write(result_path_noise, noised_waveform, sample_rate)
            st.audio(result_path_noise)

            # Traitement de segmentation
        if segmentation_option and st.button('Segmenter'):
           
            if uploaded_file is not None:
                
                # Save the uploaded file to a temporary path
                temp_file_path = "temp_uploaded_file.wav"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Clear the audios_folder before storing new data
                st.write("Clearing old data...")
                clear_folder(audios_folder)

                # Segment the audio file and get the path to the ZIP file
                zip_path = segment_audio(temp_file_path, 30, audios_folder)  # 30 secondes par segment

                # Provide the ZIP file for download
                with open(zip_path, "rb") as file:
                    st.download_button("Télécharger les segments", file.read(), file_name="segments.zip")


        if training_option and st.button('Traiter et Entraîner'):
            if uploaded_file is not None:
                
                st.write("All the audio files have been extracted.")

                # if start_training and not st.session_state.model_trained :
                st.title("Please wait, the training of your model is in progress. This may take an average of 10 minutes..")

                # Chemin du dossier existant où vous voulez stocker les fichiers audio extraits
                existing_folder = 'C:\\Users\\hanae\\Documents\\VOICE_REC_ML\\newdata'

                # Nom du dossier où vous voulez stocker les fichiers audio extraits
                audios_folder = os.path.join(existing_folder, 'extracted_audios')

                import glob

                POS = 'newdata/extracted_audios'
                NEG = 'newdata/autres'

                neg_files = glob.glob(NEG + '/**/*.wav', recursive=True)

                pos = tf.data.Dataset.list_files(POS+'/*.wav')
                neg = tf.data.Dataset.from_tensor_slices(neg_files)

                # #LABELS 
                positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(len(pos)))))
                negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(len(neg)))))
                #Mettre les pos et les neg dans la mm var 
                data = positives.concatenate(negatives)

        # #------------------------------------------------------------------------------

                lengths = []
                new_voice_path = os.path.join('newdata', 'extracted_audios')

                for file in os.listdir(new_voice_path):
                    # Skip if not a wav file
                    if not file.endswith('.wav'):
                        continue
                    full_path = os.path.join(new_voice_path, file)
                    tensor_wave = load_wav_16k_mono(full_path)
                    lengths.append(len(tensor_wave))

        # #------------------------------------------------------------------------------
        # #TRANSFORM EN SPECTO 
                def preprocess(file_path, label): 
                    # Skip if not a wav file
                    if file.endswith('.wav'):
                    # if tf.strings.regex_full_match(file_path, '.*\.wav'):
                        wav = load_wav_16k_mono(file_path)
                        wav = wav[:480000]
                        zero_padding = tf.zeros([480000] - tf.shape(wav), dtype=tf.float32)
                        wav = tf.concat([zero_padding, wav],0)
                        spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
                        spectrogram = tf.abs(spectrogram)
                        spectrogram = tf.expand_dims(spectrogram, axis=2)
                        return spectrogram, label

                filepath, label = positives.shuffle(buffer_size=10000).as_numpy_iterator().next()

                spectrogram, label = preprocess(filepath, label)

        # #------------------------------------------------------------------------------

                data = data.map(preprocess)
                data = data.cache()
                data = data.shuffle(buffer_size=100)
                data = data.batch(16)
                data = data.prefetch(8)

                train = data.take(3)
                test = data.skip(3).take(2)

                samples, labels = train.as_numpy_iterator().next()

        # #------------------------------------------------------------------------------
                # #deep model
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Conv2D, Dense, Flatten

                from keras.layers import MaxPooling2D

                model = Sequential()
                model.add(Conv2D(16, (3,3), activation='relu', input_shape=(14991, 257,1)))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Conv2D(16, (3,3), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Flatten())
                model.add(Dense(128, activation='relu'))
                model.add(Dense(1, activation='sigmoid'))

                model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

                print(model.summary())

        # #------------------------------------------------------------------------------
                # #TRAIN
                hist = model.fit(train, epochs=4, validation_data=test)
                # SAVE MODEL
                model.save('your_voice_recognition_model.h5')

                # st.session_state.model_trained = True
                # st.write("L'entraînement est terminé. Vous pouvez télécharger le modèle.")

                # if st.session_state.model_trained:
                # Chemin vers le fichier modèle
                model_path = 'your_voice_recognition_model.h5'

                # Lire le fichier modèle en tant que bytes
                with open(model_path, 'rb') as file:
                    model_bytes = file.read()

                # Créer un bouton de téléchargement pour le modèle
                st.download_button(
                    label="Télécharger votre modèle de reconnaissance vocale",
                    data=model_bytes,
                    file_name="your_voice_recognition_model.h5",
                    mime="application/octet-stream"
                )



if __name__ == "__main__":
    main()
