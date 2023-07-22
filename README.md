# autoencoder-audio
Informations to use this project.

Launch the notebook to try main functions.

You can retrieve url from videos with a specified duration, prompt, year.
using this function : search_video_youtube(language,afterdate,beforedate,json_path_file_name,developper_key,prompt,durations):

with the url retrieved you can build a df dataset. by chunking video you will need to use this function : 

generate_df_dataset(language)

with the specified language.


You will need to use your own developper key.
You need to retrieve url in order to build a datasets with youtube videos id.

Then you can train an autoencoder with youtube audio, this can take a lot of time if the mp3 are not downloaded.
