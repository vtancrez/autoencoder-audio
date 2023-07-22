import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch
import pandas as pd
import isodate
from audio_processing import download_youtube_audio, mp3_to_mel_spectrogram

def get_transformed_audio_segment(language, videoid, time_start, time_end,sr=44100,hop_length=512):
    """
    Returns a slice of preprocessed audio data from a given language video at a specified time interval. Computed with sr and hop_length information.
    
    Args:
        language (str): Language of the video.
        videoid (str): YouTube ID of the video.
        time_start (float): Start time for the slice in minutes.
        time_end (float): End time for the slice in minutes.
        sr (int, optional): Sampling rate for the audio transformation. Defaults to 44100.
        hop_length (int, optional): Hop length for the audio transformation. Defaults to 512.

    Returns:
        np.ndarray: Sliced preprocessed audio data as a numpy array.
    """
    frequency_factor=(1000/sr)*512
    np_data = np.load(f"./data_transformed/{language}_{videoid}_transformed.npy")
    np_data = np_data[round(time_start*(1000/frequency_factor)*60):round(time_end*(1000/frequency_factor)*60)]
    return np_data

def generate_dataset(language, df, indice,batch_size=64,sr=44100,hop_length=512):
    """
    Generates a DataLoader from slices of preprocessed audio data for a given range of rows in the dataframe.
    
    Args:
        language (str): Language of the videos.
        df (pd.DataFrame): DataFrame containing video information (must have 'url', 'début', and 'fin' columns).
        indice (int): Starting index for the slice of rows from the DataFrame.
        batch_size (int, optional): Number of samples per batch to load. Defaults to 64.
        step (int, optional): Number of consecutive rows from the DataFrame to process. Defaults to 5.
        sr (int, optional): Sampling rate for the audio transformation. Defaults to 44100.
        hop_length (int, optional): Hop length for the audio transformation. Defaults to 512.

    Returns:
        DataLoader: DataLoader containing the preprocessed audio data. And if download failed return None.
    
    """
    step=1
    global_datasets = None
    rows_to_take = df.iloc[indice:indice+step]

    for _, row in rows_to_take.iterrows():
        videoid = row['video_id']
        start_time = row['start']
        end_time = row['end']
        res = download_youtube_audio(language, videoid)
        if res == False:
            return None
        mp3_to_mel_spectrogram(language, videoid, sr=sr, hop_length=hop_length)

        if global_datasets is None:
            global_datasets = get_transformed_audio_segment(language, videoid, start_time, end_time,sr=sr,hop_length=hop_length)
        else:
            np_data = get_transformed_audio_segment(language, videoid, start_time, end_time,sr=sr,hop_length=hop_length)
            global_datasets = np.concatenate((global_datasets, np_data), axis=0)
    tensor_data = torch.from_numpy(global_datasets)
    tensor_dataset = TensorDataset(tensor_data) 
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=True)

    return dataloader

