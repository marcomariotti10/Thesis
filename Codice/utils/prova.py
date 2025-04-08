import torch
import torch.nn as nn
from torchsummary import summary
import torch.nn.init as init

# Define the Autoencoder_big class
class Autoencoder_big(nn.Module):
    def __init__(self, activation_fn=nn.ReLU):  # Constructor method for the autoencoder
        super(Autoencoder_big, self).__init__()  # Calls the constructor of the parent class (nn.Module) to set up necessary functionality.
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            activation_fn(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            activation_fn(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            activation_fn(),
            nn.MaxPool2d(2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            activation_fn(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            activation_fn(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            activation_fn(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            activation_fn(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(32, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):  # The forward method defines the computation that happens when the model is called with input x.
        x = self.encoder(x).contiguous()
        x = self.decoder(x).contiguous()
        return x

def initialize_weights(m):
    """Applies weight initialization to the model layers."""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):  
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  
        if m.bias is not None:
            init.constant_(m.bias, 0)  
    elif isinstance(m, nn.Linear):  
        init.xavier_normal_(m.weight)  
        if m.bias is not None:
            init.constant_(m.bias, 0)  
    elif isinstance(m, nn.BatchNorm2d):  
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)



# Instantiate the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder_big().to(device)

x = torch.rand(16, 1, 400, 400).to(device)  # Batch size of 16, 1 channel, 400x400 images

with torch.no_grad():
    # Dynamically compute encoder activations
    encoder_activations = []
    activation = x
    for layer in model.encoder:
        activation = layer(activation)
        encoder_activations.append(activation)

    # Dynamically compute decoder activations
    decoder_activations = []
    activation = encoder_activations[-1]  # Start with the last encoder activation
    for layer in model.decoder:
        activation = layer(activation)
        decoder_activations.append(activation)

# Calculate and print dead neuron percentages for encoder and decoder layers
for i, e_activation in enumerate(encoder_activations):
    print(f"Dead neurons in encoder layer {i + 1}: {dead_neuron_percentage(e_activation):.2f}%")

for i, d_activation in enumerate(decoder_activations):
    print(f"Dead neurons in decoder layer {i + 1}: {dead_neuron_percentage(d_activation):.2f}%")

model.apply(initialize_weights)  # Apply weight initialization

summary(model, (1, 400, 400))  # Print the model summary for input size (1, 400, 400)
# Generate random input data and move it to the same device

# Forward pass to get activations
with torch.no_grad():
    # Dynamically compute encoder activations
    encoder_activations = []
    activation = x
    for layer in model.encoder:
        activation = layer(activation)
        encoder_activations.append(activation)

    # Dynamically compute decoder activations
    decoder_activations = []
    activation = encoder_activations[-1]  # Start with the last encoder activation
    for layer in model.decoder:
        activation = layer(activation)
        decoder_activations.append(activation)

# Calculate and print dead neuron percentages for encoder and decoder layers
for i, e_activation in enumerate(encoder_activations):
    print(f"Dead neurons in encoder layer {i + 1}: {dead_neuron_percentage(e_activation):.2f}%")

for i, d_activation in enumerate(decoder_activations):
    print(f"Dead neurons in decoder layer {i + 1}: {dead_neuron_percentage(d_activation):.2f}%")