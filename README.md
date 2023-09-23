# autoencoder-audio

This project has been made during my internship in qarma LIS (laboritory in AI) the goal is to study how autoencoder learn audio in different languages. My tutor was Thomas Schatz.


Informations to use this project.

Launch the notebook to try main functions.

search_video_youtube : 
You can retrieve url from videos with a specified duration, prompt, year.
using this function : search_video_youtube(language,afterdate,beforedate,json_path_file_name,developper_key,prompt,durations):


generate_df_dataset : 
with the url retrieved you can build a df dataset with the specified language. by chunking video you will need to use this function : 
generate_df_dataset(language)


training_autoencoder :
a function that take an autoencoder model from the module autoencoder.py, a df dataset train validation and test defined by the function above, it also takes as arguments le number of lines that will be used for each dataset (train, validation,test)
training_autoencoder(model,df_train,df_validation,df_test,language=language,val_len=8,test_len=8,nb_data_max=32)
you can build df_train, df_validation,df_test on your own by filling those columns : chunk (whatever) , lang (language of the videos), video_id ( the youtube video id you want to use for training, testing), start, end.

start is the beggining of the slice we take from the videos.
end is the upper limit of the slice we take from videos.


Training an autoencoder can take a lot of times depeding of how is the architecture, how much data you have and if the video are not yet downloaded.


