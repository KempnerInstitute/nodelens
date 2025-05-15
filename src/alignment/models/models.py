from torch import nn

class MLP(nn.Module):
    """
    3 hidden layer fully-connected relu network for MNIST including dropouts after input layer

    includes optional kwargs for changing the hidden layer widths and the input dimensionality
    """

    def __init__(self, input_dim=784, hidden_widths=[100, 100, 50], output_dim=10, dropout=0.5, linear=False):
        super().__init__()
        
        """architecture definition"""

        # Define activation function (ReLU is default, but network can be "LINEAR" if requested)
        if linear:
            activation = nn.Identity()
        else:
            activation = nn.ReLU()

        # Input layer is always Linear then ReLU
        self.layerInput = nn.Sequential(nn.Linear(input_dim, hidden_widths[0]), activation)

        # Hidden layers are dropout / Linear / ReLU
        self.layerHidden = nn.ModuleList()
        for ii in range(len(hidden_widths) - 1):
            hwin, hwout = hidden_widths[ii], hidden_widths[ii + 1]
            self.layerHidden.append(nn.Sequential(nn.Dropout(p=dropout), nn.Linear(hwin, hwout), activation))

        # Output layer is alwyays Dropout then Linear
        self.layerOutput = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(hidden_widths[-1], output_dim))
    
    def forward(self, x):
        x = self.layerInput(x)
        for hidden_layer in self.layerHidden:
            x = hidden_layer(x)
        return self.layerOutput(x)

class CNN2P2(nn.Module):
    """
    CNN with 2 convolutional layers, a max pooling stage, and 2 feedforward layers with dropout

    Has flexible build method but strict 2 convolutional / max pooling / 2 feedforward architecture
    """

    def __init__(
        self,
        in_channels=1,
        output_dim=10,
        channels=[32, 64],
        kernel_size=[5, 5],
        stride=[1, 1],
        padding=[2, 2],
        num_hidden=[3136, 128],
        dropout=0.5,
        flag=True,
    ):
        super().__init__()
        """architecture definition"""

        for val, name in zip(
            (channels, kernel_size, stride, padding),
            ("channels", "kernel_size", "stride", "padding"),
        ):
            assert len(val) == 2, f"{name} must be 2 elements describing the parameter for each of two convolutional layers"
        assert len(num_hidden) == 2, "num_hidden must be 2 elements describing the number of hidden for each of two feedforward layers"

        # create layers
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                channels[0],
                kernel_size=kernel_size[0],
                stride=stride[0],
                padding=padding[0],
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                channels[0],
                channels[1],
                kernel_size=kernel_size[1],
                stride=stride[1],
                padding=padding[1],
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Flatten(start_dim=1),
        )
        self.layer3 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_hidden[0], num_hidden[1]),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(num_hidden[1], output_dim),
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)