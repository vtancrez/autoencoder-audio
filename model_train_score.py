from torch import nn
import torch
from create_data_set import generate_dataset
import pandas as pd
def train_model(model, loader, loss_fn, optimizer, device):
    """
    Train a given model for one epoch. Each batch from the loader is processed 
    to update the model parameters.
    
    Args:
        model (nn.Module): The model to be trained.
        loader (DataLoader): Loader for the training data.
        loss_fn (nn.Module): The loss function used for training.
        optimizer (Optimizer): The optimizer used for training.
        device (torch.device): The device on which to perform training.
    
    Returns:
        nn.Module: The trained model.
    """
    
    model.train()
    for batch in loader:
        batch = batch[0].float().to(device)
        optimizer.zero_grad()
        _, outputs = model(batch)
        loss = loss_fn(outputs, batch)
        loss.backward()
        optimizer.step()
    return model

def iterates_on_loader(model,optimizer, loss_fn, device,language,df,nb_data,train=False):
    """
    Iterates over the data loader either for training or for computing loss depending on the 'train' argument.

    Args:
        model (nn.Module): The model to be trained or evaluated.
        optimizer (Optimizer): The optimizer used for training.
        loss_fn (nn.Module): The loss function used for training.
        device (torch.device): The device on which to perform computations.
        language (str): Language of the data.
        df (DataFrame): Input data.
        nb_data (int): Number of data points to consider.
        step (int, optional): Number of data points to consider in each iteration. Default is 4.
        train (bool, optional): If True, perform training, else perform evaluation. Default is False.
    
    Returns:
        If train is True, returns the trained model (nn.Module). 
        If train is False, returns the average loss per batch (float).
        lines_df (df): lines of the dataframe used for training or testing
    """
    indice=0  
    step=1
    lines_used=pd.DataFrame([])
    if train:
        while indice<nb_data:
            if indice+step>nb_data:
                step=nb_data-indice
            train_dataloader=generate_dataset(language,df,indice)
            if train_dataloader!=None:
                model = train_model(model, train_dataloader, loss_fn, optimizer, device)
                lines_used = pd.concat([lines_used,df.iloc[indice:indice+step]], ignore_index=True)
            else:
                nb_data+=1
            indice+=step
        return model,lines_used    
    else:  
        train_loss = 0
        total_batches=0
        while indice<nb_data:
            if indice+step>nb_data:
                step=nb_data-indice
            train_dataloader=generate_dataset(language,df,indice)
            if train_dataloader!=None:
                total_batches+=len(train_dataloader)
                train_loss += score(model, train_dataloader, loss_fn, device)
                lines_used = pd.concat([lines_used,df.iloc[indice:indice+step]], ignore_index=True)
            else:
                nb_data+=1
            indice+=step
        return train_loss/total_batches,lines_used
def score(model, loader, loss_fn, device):
    """
    Computes the total loss over all batches in a data loader.

    Args:
        model (nn.Module): The model to be evaluated.
        loader (DataLoader): The loader for the data.
        loss_fn (nn.Module): The loss function.
        device (torch.device): The device on which to perform computations.
    
    Returns:
        float: Total loss.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch[0].float().to(device)
            _, outputs = model(batch)
            loss = loss_fn(outputs, batch)
            total_loss += loss.item()
    return total_loss
  
def training_autoencoder(model,df_train,df_validation,df_test,language,nb_data_max=32,val_len=8,test_len=8,device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    """
    Trains an autoencoder on the training data, while also evaluating it on the validation and test data.

    Args:
        model (nn.Module): The autoencoder model to be trained.
        df_train (DataFrame): Training data.
        df_validation (DataFrame): Validation data.
        df_test (DataFrame): Test data.
        language (str): Language of the data.
        val_len (int, optional): Length of validation data. Default is 8.
        test_len (int, optional): Length of test data. Default is 8.
        device (torch.device, optional): The device on which to perform computations. Default is GPU if available, else CPU.
    
    Returns:
        list: Lists containing the training, validation and test losses and the number of epochs.
        Dataframe: df containing the lines used during the training/validation/test.
    """
    nb_data=1
    loss_test=[]
    loss_train=[]
    loss_val=[]
    number_epochs=[]
    while nb_data<=nb_data_max:
        model_iteration=model
        loss_fn = nn.MSELoss()
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3)
        print(f"nb data : {nb_data}")
        best_model=model
        epoch=0
        best_val_loss = float("inf")
        patience_indice=0 # increase by 1 each time the model do worse than previously 
        while patience_indice<=2:
            epoch+=1
            print(f"epoch : {epoch} ")
            model_iteration,_=iterates_on_loader(model_iteration,optimizer, loss_fn, device,language,df_train,nb_data,train=True)
            val_loss,lines_used_validation= iterates_on_loader(model_iteration,optimizer, loss_fn, device,language,df_validation,val_len,train=False)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model=model
                patience_indice=0
            else:
                patience_indice+=1
        print("train loss : ")
        number_epochs.append(epoch)
        loss_for_train,lines_used_training=iterates_on_loader(best_model,optimizer, loss_fn, device,language,df_train,nb_data,train=False)
        loss_train.append(loss_for_train)
        loss_val.append(best_val_loss)
        loss_for_test,lines_used_test=iterates_on_loader(best_model,optimizer, loss_fn, device,language,df_test,test_len,train=False)
        loss_test.append(loss_for_test)
        nb_data*=2
    return loss_train,loss_val,loss_test,number_epochs,lines_used_training,lines_used_validation,lines_used_test