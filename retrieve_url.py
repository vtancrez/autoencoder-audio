from googleapiclient.errors import HttpError
from googletrans import Translator
from googleapiclient.discovery import build
import ujson as json
from tqdm import tqdm
import os
def get_api(developper_key):
    """
    Constructs and returns a YouTube Data API instance with the specified API service name, 
    version, and developer key.

    a Developer Key is required to use the YouTube Data API.
    Args:
        developper_key (str): the developper key to use
    Returns:
        object: The built YouTube Data API instance.
    """
    DEVELOPER_KEY=developper_key
    YOUTUBE_API_SERVICE_NAME = 'youtube'
    YOUTUBE_API_VERSION = 'v3'
    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
                    developerKey=DEVELOPER_KEY)
    return youtube

def search(youtube, **kwargs):
    """
    Searches for videos on Youtube using the provided YouTube Data API and search parameters.

    Args:
        youtube (object): The YouTube Data API instance.
        **kwargs (dict): Arbitrary keyword arguments. These represent the search parameters. 
    Returns:
        dict: The response from the YouTube API containing the search results. The structure 
              of the return is defined by the API and depends on the 'part' parameter in kwargs.
    """
    search_response = youtube.search().list(**kwargs).execute()
    return search_response



langs = ["afrikaans", "arabic", "chinese", "croatian", "czech", "german", "estonian", "english", "spanish", "french", 
         "greek", "hebrew", "korean", "hungarian", "icelandic","indonesian", "hindi", "italian", "japanese", "lithuanian",
         "malay", "dutch", "maori", "polish", "portuguese", "romanian", "swedish", "telugu", "turkish"]


# languages names in English to avoid Unicode complications
# available MBROLA voices without Iranian as it doesn't have a ISO-639-1 code, and breton which is not supported by googletrans,
# and latin which has no query result on YouTube
# in total 29 languages
# ISO-639-1 code, see http://www.loc.gov/standards/iso639-2/php/code_list.php 


with open("json/langs_id.json") as file:
    lang_ids = json.load(file)

def get_video_details(youtube, video_id):
    # Call the API's videos.list method to retrieve video details.
    results = youtube.videos().list(
        part="contentDetails", # Part signifies that we need contentDetails 
        id=video_id  # video id
    ).execute()
    return results['items'][0]['contentDetails'] if results['items'] else None

def search_video_youtube_languages(languages,year,developper_key,prompt,duration):
    
    """ search of the youtube url corresponding to the list of languages and the prompt with the developper key put in argument.
    
    Args : 
        languages(list of string): a list of languages
        year (int): year of the search
        developper_key (str): developper key to use.
        prompt (str): prompt for the search.
        duration (tab of string): containing the duration of videos : "long", "medium" or "short"
    """
    date_after=f"{year}-01-01T00:00:00Z"
    date_before=f"{year}-12-31T23:59:59Z"
    for language in languages:
        if not os.path.exists(f"informations_videos/{language}"):
            os.makedirs(f"informations_videos/{language}")
        if os.path.exists(f"informations_videos/{language}/{year}_{language}.json"):
            print("A json file for this language and year already exists. Skipping...")
            continue
        search_video_youtube(language,date_after,date_before,f"informations_videos/{language}/{year}_{language}.json",developper_key,prompt,duration)

def search_video_youtube(language,afterdate,beforedate,json_path_file_name,developper_key,prompt,durations):
    """
    Searches for long duration 'prompt videos in a specific language within a given date range 
    on YouTube and saves the search results in a JSON file. The search is done using the YouTube Data API.

    Args:
        language (str): The language to search for the videos in.
        afterdate (str): Videos uploaded after this date (in RFC 3339 format: YYYY-MM-DDTHH:MM:SSZ) will be included in the search.
        beforedate (str): Videos uploaded before this date (in RFC 3339 format: YYYY-MM-DDTHH:MM:SSZ) will be included in the search.
        json_path_file_name (str): where we the json file will be put.
        developper_key (str): developper key to use.
        prompt (str): prompt of the search
        duration (tab of string): containing the duration of videos : "long", "medium" or "short"
    Notes:
        Daily quota for YouTube API is 100 search pages. Each page contains at most 50 results, which is the maximum allowed.
    Returns:
        None. But writes the search result into a json file named as <language>.json in the current directory.
    """
    #the daily quota is 100 search pages
    maxResults = 50  # max allowed by page
    part = 'id, snippet'
    item_type = 'video'
    #durations=["long", "medium", "short"]
    translator = Translator()
    youtube = get_api(developper_key)
    langs=[language]
    res = {lang: [] for lang in langs}
    lang_qs = {lang: translator.translate(prompt, dest=lang_ids[lang]).text for lang in langs}
    nb_pages_per_lang=8
    for lang in langs:
        for duration in durations:
            results=True
            page=0
            while results and page<=nb_pages_per_lang:
                try:
                    if page == 0:
                        res[lang].append(search(youtube, maxResults=maxResults, part=part, type=item_type, q=lang_qs[lang], 
                                                videoDuration=duration,
                                                publishedAfter=afterdate,publishedBefore=beforedate,safeSearch="strict",relevanceLanguage=lang_ids[lang]))
                    else:
                        if res[lang] and 'nextPageToken' in res[lang][-1]:
                            pageToken = res[lang][-1]['nextPageToken']
                            res[lang].append(search(youtube, pageToken=pageToken, maxResults=maxResults, part=part,
                                                    type=item_type, q=lang_qs[lang], videoDuration=duration,publishedAfter=afterdate,
                                                    publishedBefore=beforedate,safeSearch="strict",relevanceLanguage=lang_ids[lang]))
                        else:
                            print(f'No more results for {lang}. Numbers of pages browsed : {page} for video duration : {duration}')
                            results=False
                     # get video details for each search result
                    for item in res[lang][-1]['items']:
                        video_id = item['id']['videoId']
                        video_details = get_video_details(youtube, video_id)
                        if video_details:
                            item['contentDetails'] = video_details
                except HttpError as e:
                    print(f'An HTTP error occurred for {lang}: {e.resp.status}: {e.content}')
                page+=1

    with open(f"{json_path_file_name}", 'w') as f:
        json.dump(res, f)
        
             
def create_json_url(language,afterdate,beforedate):
    """
    Generates a JSON file containing the video IDs of 'children cartoon' videos in a specific language 
    within a given date range on YouTube. The function first performs a search operation via 'search_video_youtube()' 
    function (if required JSON file does not exist), and then filters the results based on the detected language of 
    the video's title and description.

    Args:
        language (str): The language to search for the videos in.
        afterdate (str): Videos uploaded after this date (in RFC 3339 format: YYYY-MM-DDTHH:MM:SSZ) will be included in the search.
        beforedate (str): Videos uploaded before this date (in RFC 3339 format: YYYY-MM-DDTHH:MM:SSZ) will be included in the search.

    Notes:
        The function also uses the Google Translate API to detect the language of the video's title and description. This filter is useful 
        because search() function does not always return videos in the specified language.
        The resultant JSON file is saved in a directory named 'videos' in the current directory.

    Returns:
        None. But writes the video IDs into a json file named as videos_<language>.json in the ./videos directory.
    """
    if not os.path.isfile(f"{language}.json"):
        search_video_youtube(language,afterdate,beforedate)
    with open(f"{language}.json", 'r') as f:
            res = json.load(f)    
            
    if not os.path.exists("./videos"):
        os.mkdir("./videos")
    translator = Translator()
    # on ethernet it took 3min for 1400 différents videoid
    lang=language
    video_dict = {lang: []}
    for search in res[lang]:
        print(search)
        for item in search["items"]:
            # we further filter search results by inspecting the language of title and description  
            if translator.detect(item["snippet"]["title"]).lang == lang_ids[lang] and translator.detect(item["snippet"]["description"]).lang == lang_ids[lang]:
                video_dict[lang].append(item["id"]["videoId"])
    with open(f"./videos/videos_{lang}.json", 'w') as g:
        json.dump(video_dict, g)
        
    del video_dict
  
  
def check_number_url(langs,year):
    for lang in langs:
        if not os.path.exists(f"informations_videos/{lang}/{year}_{lang}.json"):
            print("no json files for languae : ",lang)
            continue
        with open(f"informations_videos/{lang}/{year}_{lang}.json", 'r') as f:
            data = json.load(f)
            video_dict=[]
            for search in data[lang]:  
                for item in search["items"]:
                    video_dict.append(item["id"]["videoId"])
            if (len(video_dict)==0):
                print(f"no videos for language : {lang}")
            #print(f"languages : {lang}, year : {year}, number of videos : {len(video_dict)}")
            