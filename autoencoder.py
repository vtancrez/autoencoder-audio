from torch import nn

class AutoEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        """
        Constructs all the necessary attributes for the AutoEncoder object.

        Parameters:
            input_size (int): Size of the input vectors.
            hidden_size (int): Size of the hidden layers in the encoder and decoder.
            num_layers (int): Number of layers in the encoder and decoder.
        """
        super().__init__()

        assert num_layers >= 1, "Number of layers must be at least 1"

        # Encoder
        encoder_layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 1):  
            encoder_layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        for _ in range(num_layers - 1):  
            decoder_layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
        decoder_layers.append(nn.Linear(hidden_size, input_size)) 
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        """
        Performs a forward pass of the input through the autoencoder.

        Parameters:
            x (Tensor): The input to the autoencoder.
        
        Returns:
            encoding (Tensor): The reduced representation of the input.
            reconstructed_x (Tensor): The reconstructed input.
        """
        encoding = self.encoder(x)
        reconstructed_x = self.decoder(encoding)
        return encoding, reconstructed_x
