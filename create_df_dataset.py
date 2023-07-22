import os
import isodate
import json
import pandas as pd

# Load JSON data
def load_video_data(year, language):
    """
    Load JSON data for a given year and language.

    Args:
        year (int): The year for which data is to be loaded.    if os.path.exists(f"./data_transformed/{language}_{videoid}_transformed.npy"):
        #print(f"npy transformed alreay exist for {videoid}")
        return
    
        language (str): The language for which data is to be loaded.

    Returns:
        dict or None: A dictionary containing the loaded JSON data if the file exists, None otherwise.
    """
    if os.path.exists(f"data/{language}/{year}_{language}.json"):
        with open(f"data/{language}/{year}_{language}.json", 'r') as f:
            data = json.load(f)
            return data
    return None

def filter_and_process_video_data(data, language, max_duration,df):
    """
    Process the loaded JSON data by filtering and extracting relevant information.

    Args:
        data (dict): The loaded JSON data.
        language (str): The language of the video data.
        max_duration (int): The maximum duration (in seconds) for a video to be included in the dataset.

    Returns:
        DataFrame: A pandas DataFrame containing the processed video data.
    """
    for search in data[language]:
        for item in search["items"]:
            duration_str = item["contentDetails"]["duration"]
            islive = item["snippet"]["liveBroadcastContent"]
            if (islive=="none"):
                duration = isodate.parse_duration(duration_str)
                if duration.total_seconds() < max_duration:
                    df = extract_video_data(item, duration, language,df)
    return df

def extract_video_data(item, duration, language,df):
    """
    Process an individual video item by extracting relevant information and appending to the dataframe.

    Args:
        item (dict): A dictionary representing a single video item.
        duration (timedelta): The duration of the video item.
        language (str): The language of the video.

    Returns:
        DataFrame: A pandas DataFrame containing the processed video data.
    """
    videoid = item["id"]["videoId"]
    duration_minutes = duration.total_seconds() / 60  # Convert to minutes
    five_minutes_slice = duration_minutes // 5
    time_start = ((duration_minutes % 5)) / 2
    num_chunk=1
    for i in range(int(five_minutes_slice)):
        new_row = {'chunk':num_chunk,'lang':language,'video_id': videoid, 'start': time_start + i * 5, 'end': time_start + (i+1) * 5}
        new_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_df])
        num_chunk+=1
    return df

def generate_df_dataset(language, start_year=None, end_year=None, max_duration=3600):
    """
    Create a video dataset for a given language and duration constraint, across a range of years.

    Args:
        language (str): The language of the video data.
        start_year (int, optional): The start year for the data collection. Default is 2010.
        end_year (int, optional): The end year for the data collection. Default is 2023.
        max_duration (int, optional): The maximum duration (in seconds) for a video to be included in the dataset. Default is 3600.

    Returns:
        DataFrame: A pandas DataFrame containing the video dataset.
    """
    dataset = {}
    df = pd.DataFrame(columns=["chunk","lang","video_id","start", "end"])
    if(start_year==None):
        start_year = 2010
    if(end_year==None):
        end_year = 2023
    for year in range(start_year, end_year+1):
        data = load_video_data(year, language)
        if data:
            df = filter_and_process_video_data(data, language, max_duration,df)
    df = df.sample(frac=1).reset_index(drop=True)
    return df

