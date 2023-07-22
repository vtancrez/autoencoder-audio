from create_data_set import *
from autoencoder import AutoEncoder
from model_train_score import *
import time 
import pandas as pd
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using {device}.")

def calculate_hidden_size(m, D, L):
    """
    Calculate the hidden size 'n' for the autoencoder model based on the number of model parameters 'm', 
    input size 'D', and number of layers 'L'. 

    Args:
        m (int): The number of model parameters.
        D (int): The input size.
        L (int): The number of layers.

    Returns:
        int: The hidden size 'n'.
        int: The total number of parameters.
    """
    n = 0
    while True:
        val = 2*n*D + n + D + 2*(n**2 + n)*(L - 1)
        if val >= m:
            return n,val
        n += 1
        
def create_autoencoder_loss_df(df_train,df_validation,df_test,language, L_max=3, D=128, initial_parameters=4096,nb_data_max=32,val_len=8,test_len=8):
    """
    Generate a dataframe containing the loss from training, validation, and testing on an autoencoder model.

    Args:
        df_train (DataFrame): Training data.
        test_dataloader (DataLoader): DataLoader for test data.
        validation_dataloader (DataLoader): DataLoader for validation data.
        L_max (int, optional): Maximum number of layers. Default is 3.
        D (int, optional): Input size. Default is 128.
        initial_parameters (int, optional): Initial number of parameters. Default is 2000.

    Returns:
        DataFrame: A dataframe containing the losss, number of data points, number of layers, number of parameters, and the evaluation set.
    """
    data = []

    for num_layers in range(1, L_max+1):
        parameters = initial_parameters
        for param in range(4):
            print(f"Training for {parameters} parameters.")
            print(num_layers)
            hidden_size, val = calculate_hidden_size(parameters, D, L_max)
            model = AutoEncoder(input_size=D, hidden_size=hidden_size, num_layers=num_layers).to(device)
            start_time=time.time()
            train_loss, val_loss, test_loss,number_epoch,lines_used_train,lines_used_validation,lines_used_test = training_autoencoder(model,df_train,df_validation,df_test,language=language,val_len=val_len,test_len=test_len,nb_data_max=nb_data_max)
            lines_used_train.to_csv(f"df_used_for_training_testing/language_{language}_parameters_{parameters}_numlayers_{num_layers}_train.csv")
            lines_used_validation.to_csv(f"df_used_for_training_testing/language_{language}_parameters_{parameters}_numlayers_{num_layers}_validation.csv")
            lines_used_test.to_csv(f"df_used_for_training_testing/language_{language}_parameters_{parameters}_numlayers_{num_layers}_test.csv")
            end_time=time.time()
            duration=end_time-start_time
            # Assuming loss arrays are the same size
            for idx in range(len(train_loss)):
                data.extend([
                    {"loss": train_loss[idx], "training_set_id": 0, "number_data": 2**idx, "number_layers": num_layers, "number_parameters_ideal": parameters,"number_parameters_effective": val, "eval_set": "train","duration":duration,"number_epoch":number_epoch[idx],"language":language},
                    {"loss": val_loss[idx], "training_set_id": 0, "number_data": 2**idx, "number_layers": num_layers, "number_parameters_ideal": parameters,"number_parameters_effective": val, "eval_set": "val","duration":duration,"number_epoch":number_epoch[idx],"language":language},
                    {"loss": test_loss[idx], "training_set_id": 0, "number_data": 2**idx, "number_layers": num_layers, "number_parameters_ideal": parameters,"number_parameters_effective": val, "eval_set": "test","duration":duration,"number_epoch":number_epoch[idx],"language":language}
                ])
            parameters *= 2
    df = pd.DataFrame(data)
    return df

