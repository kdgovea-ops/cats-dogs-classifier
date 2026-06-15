import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self, 
                 num_conv_layers=3,
                 conv_channels=[32, 64, 128],
                 kernel_sizes=[3, 3, 3],
                 pool_type='max',  # 'max' or 'avg'
                 dropout_rate=0.5,
                 num_hidden_layers=2,
                 hidden_sizes=[256, 128],
                 activation='relu'):  # 'relu', 'elu', 'leaky_relu'
        
        super(CustomCNN, self).__init__()
        
        # Store activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        else:
            self.activation = nn.ReLU()
        
        # Store pooling type
        self.pool_type = pool_type
        
        # Convolutional layers
        self.conv_layers = nn.ModuleList()
        in_channels = 3  # RGB images
        
        for i in range(num_conv_layers):
            out_channels = conv_channels[i]
            kernel_size = kernel_sizes[i]
            
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1),
                self.activation,
                nn.BatchNorm2d(out_channels)
            ))
            in_channels = out_channels
        
        # Pooling layer (same for all)
        if pool_type == 'max':
            self.pool = nn.MaxPool2d(2, 2)
        else:  # avg
            self.pool = nn.AvgPool2d(2, 2)
        
        # Calculate flattened size after convolutions and pooling
        # After num_conv_layers, image is divided by 2^num_conv_layers
        # 224x224 -> after pooling: 224 / (2^num_conv_layers)
        final_size = 224 // (2 ** num_conv_layers)
        flattened_size = conv_channels[-1] * final_size * final_size
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc_layers = nn.ModuleList()
        in_size = flattened_size
        
        for hidden_size in hidden_sizes:
            self.fc_layers.append(nn.Sequential(
                nn.Linear(in_size, hidden_size),
                self.activation,
                self.dropout
            ))
            in_size = hidden_size
        
        # Output layer (2 classes: cat, dog)
        self.fc_out = nn.Linear(in_size, 2)
    
    def forward(self, x):
        # Convolutional layers with pooling
        for conv in self.conv_layers:
            x = conv(x)
            x = self.pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        for fc in self.fc_layers:
            x = fc(x)
        
        # Output
        x = self.fc_out(x)
        return x