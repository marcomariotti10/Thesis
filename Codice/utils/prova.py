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

# Function to calculate dead neuron percentage
def dead_neuron_percentage(activations):
    # activations: (batch_size, num_neurons, height, width)
    print(activations.shape)
    num_neurons = activations.shape[1] * activations.shape[2] * activations.shape[3]
    # For each neuron, check if it was always zero across the batch
    dead_neurons = (activations == 0).all(dim=(0, 2, 3)).sum().item()
    return 100.0 * dead_neurons / activations.shape[1]

# Instantiate the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder_big().to(device)

model.apply(initialize_weights)  # Apply weight initialization

summary(model, (1, 400, 400))  # Print the model summary for input size (1, 400, 400)
# Generate random input data and move it to the same device
x = torch.rand(16, 1, 400, 400).to(device)  # Batch size of 16, 1 channel, 400x400 images

# Forward pass to get activations
with torch.no_grad():
    e1 = model.encoder[1](model.encoder[0](x))  # First activation in encoder
    e2 = model.encoder[4](model.encoder[3](model.encoder[2](e1)))  # Second activation in encoder
    e3 = model.encoder[7](model.encoder[6](model.encoder[5](e2)))  # Third activation in encoder

    d1 = model.decoder[1](model.decoder[0](e3))  # First activation in decoder
    d2 = model.decoder[4](model.decoder[3](model.decoder[2](d1)))  # Second activation in decoder
    d3 = model.decoder[7](model.decoder[6](model.decoder[5](d2)))  # Third activation in decoder

# Calculate and print dead neuron percentages
print(f"Dead neurons in encoder layer 1: {dead_neuron_percentage(e1):.2f}%")
print(f"Dead neurons in encoder layer 2: {dead_neuron_percentage(e2):.2f}%")
print(f"Dead neurons in encoder layer 3: {dead_neuron_percentage(e3):.2f}%")
print(f"Dead neurons in decoder layer 1: {dead_neuron_percentage(d1):.2f}%")
print(f"Dead neurons in decoder layer 2: {dead_neuron_percentage(d2):.2f}%")
print(f"Dead neurons in decoder layer 3: {dead_neuron_percentage(d3):.2f}%")