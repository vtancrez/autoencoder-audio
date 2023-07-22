from pytube import YouTube
from moviepy.editor import *
import pydub
import numpy as np
import os
import librosa
from sklearn.model_selection import train_test_split
import datasets
import pandas as pd
import isodate
import soundfile as sf
from pydub import AudioSegment
import yt_dlp


# the speed of video processing: ~80 videos per hour
# inspired from https://stackoverflow.com/questions/53633177/how-to-read-a-mp3-audio-file-into-a-numpy-array-save-a-numphttps://github.com/yt-dlp/yt-dlpy-array-to-mp3
# we can also try y, sr = librosa.load(mp3_file) and see which one is more efficient (test if normalizing affects the final mel-spectrogram)
def mp3_to_numpy(f):
    """
    Converts an MP3 file to a numpy array.
    
    Args:
        f (str): The path to the MP3 file.
    
    Returns:
        np.ndarray: The MP3 data converted to a numpy array.
    """
    a = pydub.AudioSegment.from_mp3(f).set_channels(1)
    y = np.array(a.get_array_of_samples())
    return np.float32(y) / 2**15


def numpy_to_mp3(np_array, sample_rate, filename):
    """
    Converts a numpy array to an MP3 file.
    
    Args:
        np_array (np.ndarray): The numpy array to convert to MP3.
        sample_rate (int): The sample rate of the audio data.
        filename (str): The name of the output MP3 file.
    """
    sf.write('temp.wav', np_array, sample_rate)
    audio = AudioSegment.from_wav('temp.wav')
    audio.export(filename, format='mp3')
    os.remove('temp.wav')


def mel_to_audio(mel_spectrogram, sr=44100, n_fft=2048, hop_length=512, n_iter=32):
    """
    Converts a Mel spectrogram to an audio signal. Can be useful to check if the melspectogram kept the information.
    
    Args:
        mel_spectrogram (np.ndarray): The Mel spectrogram to convert to audio.
        sr (int, optional): The sampling rate. Defaults to 44100.
        n_fft (int, optional): The number of Fourier components. Defaults to 2048.
        hop_length (int, optional): The number of samples between successive frames. Defaults to 512.
        n_iter (int, optional): The number of iterations for Griffin-Lim. Defaults to 32.
    
    Returns:
        np.ndarray: The audio signal.
    """
    spectrogram = librosa.feature.inverse.mel_to_linear(mel_spectrogram, sr=sr, n_fft=n_fft, power=1.0)

    audio = librosa.griffinlim(spectrogram, n_iter=n_iter, hop_length=hop_length)
    
    return audio

import yt_dlp

def download_youtube_audio(language, video_id):
    """
    Downloads the audio from a YouTube video and saves it as an MP3 file in the mp3 folder.
    
    Args:
        language (str): The language of the YouTube video.
        video_id (str): The ID of the YouTube video.
        
    returns:
        bool: True if the download was successful, False otherwise.
    """
    output_path = f"mp3/{language}_{video_id}.mp3"
    url=f"https://www.youtube.com/watch?v={video_id}"
    if os.path.exists(output_path):
        # print(f"mp3 already exists for {video_id}")
        return

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{video_id}.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except yt_dlp.DownloadError:
        print(f"Une erreur s'est produite lors du téléchargement de la vidéo {video_id}")
        return False

    # Move the .mp3 file to the correct directory
    os.rename(f"{video_id}.mp3", output_path)
    return True

    
        
def compute_mel_spectrogram(example, sr=44100, hop_length=512):
    """
    Computes a Mel spectrogram from audio data.
    
    Args:
        example (list or np.ndarray): The audio data.
        sr (int, optional): The sampling rate. Defaults to 44100.
        hop_length (int, optional): The number of samples between successive frames. Defaults to 512.
    
    Returns:
        np.ndarray: The Mel spectrogram.
    """
    # Access the audio data within the dictionary
    data = example
    # Convert the data to a floating-point NumPy array
    np_data = np.array(data, dtype=np.float32)
    processed_data = np.zeros((0, 128))
    mel_spectrogram = librosa.feature.melspectrogram(y=np_data, sr=sr,hop_length=hop_length, n_mels=128)
    processed_data = np.concatenate((processed_data, mel_spectrogram.T), axis=0)
    return processed_data


def mp3_to_mel_spectrogram(language,videoid,sr=44100, hop_length=512):
    """
    Converts an MP3 file to a Mel spectrogram and saves it as a numpy file.
    
    Args:
        language (str): The language of the YouTube video.
        videoid (str): The YouTube ID of the video.
        sr (int, optional): The sampling rate. Defaults to 44100.
        hop_length (int, optional): The number of samples between successive frames. Defaults to 512.
    
    Returns:
        np.ndarray: The Mel spectrogram.
    """
    if os.path.exists(f"./data_transformed/{language}_{videoid}_transformed.npy"):
        #print(f"npy transformed alreay exist for {videoid}")
        return
    
    mp3_path = os.path.join(f"mp3/{language}_{videoid}.mp3")
    x = mp3_to_numpy(mp3_path)
    # Apply transform
    transformed_data = compute_mel_spectrogram(x, sr=sr, hop_length=hop_length)
    np_data = np.concatenate([transformed_data], axis=0)
    np.save(f"./data_transformed/{language}_{videoid}_transformed.npy",np_data)
    return np_data
