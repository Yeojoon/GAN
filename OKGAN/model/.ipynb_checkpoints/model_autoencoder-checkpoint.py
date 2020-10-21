import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


IMAGE_SIZE = 784
IMAGE_WIDTH = IMAGE_HEIGHT = 28



class autoencoder(nn.Module):
    def __init__(self, input_dim=28*28, code_dim=32):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, code_dim),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(code_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, input_dim), 
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    


class conv_autoencoder_with_tanh(nn.Module):
    
    def __init__(self, code_size):
        super(conv_autoencoder_with_tanh, self).__init__()
        self.code_size = code_size
        
        # Encoder specification
        self.enc_cnn_1 = nn.Conv2d(1, 10, kernel_size=5)
        self.enc_cnn_2 = nn.Conv2d(10, 20, kernel_size=5)
        self.enc_linear_1 = nn.Linear(4 * 4 * 20, 50)
        self.enc_linear_2 = nn.Linear(50, self.code_size)
        
        # Decoder specification
        self.dec_linear_1 = nn.Linear(self.code_size, 160)
        self.dec_linear_2 = nn.Linear(160, IMAGE_SIZE)
        
    def forward(self, images):
        code = self.encode(images)
        out = self.decode(code)
        return out, code
    
    def encode(self, images):
        code = self.enc_cnn_1(images)
        code = F.relu(F.max_pool2d(code, 2))
        
        code = self.enc_cnn_2(code)
        code = F.relu(F.max_pool2d(code, 2))
        
        code = code.view([images.size(0), -1])
        code = F.relu(self.enc_linear_1(code))
        code = self.enc_linear_2(code)
        code = nn.Tanh()(code)
        return code
    
    def decode(self, code):
        out = F.relu(self.dec_linear_1(code))
        out = nn.Tanh()(self.dec_linear_2(out))
        out = out.view([code.size(0), 1, IMAGE_WIDTH, IMAGE_HEIGHT])
        return out    

    
    
class conv_autoencoder_with_sigmoid(nn.Module):
    
    def __init__(self, code_size):
        super(conv_autoencoder_with_sigmoid, self).__init__()
        self.code_size = code_size
        
        # Encoder specification
        self.enc_cnn_1 = nn.Conv2d(1, 10, kernel_size=5)
        self.enc_cnn_2 = nn.Conv2d(10, 20, kernel_size=5)
        self.enc_linear_1 = nn.Linear(4 * 4 * 20, 50)
        self.enc_linear_2 = nn.Linear(50, self.code_size)
        
        # Decoder specification
        self.dec_linear_1 = nn.Linear(self.code_size, 160)
        self.dec_linear_2 = nn.Linear(160, IMAGE_SIZE)
        
    def forward(self, images):
        code = self.encode(images)
        out = self.decode(code)
        return out, code
    
    def encode(self, images):
        code = self.enc_cnn_1(images)
        code = F.selu(F.max_pool2d(code, 2))
        
        code = self.enc_cnn_2(code)
        code = F.selu(F.max_pool2d(code, 2))
        
        code = code.view([images.size(0), -1])
        code = F.selu(self.enc_linear_1(code))
        code = self.enc_linear_2(code)
        return code
    
    def decode(self, code):
        out = F.selu(self.dec_linear_1(code))
        out = F.sigmoid(self.dec_linear_2(out))
        out = out.view([code.size(0), 1, IMAGE_WIDTH, IMAGE_HEIGHT])
        return out
    
    

class autoencoder_GMMN(nn.Module):
    def __init__(self, input_dim=28*28, code_dim=32):
        super(autoencoder_GMMN, self).__init__()
        self.encoder_fc1 = nn.Linear(input_dim, 1024)
        self.encoder_fc2 = nn.Linear(1024, code_dim)

        self.decoder_fc1 = nn.Linear(code_dim, 1024)
        self.decoder_fc2 = nn.Linear(1024, input_dim)

    def forward(self, x):
        e = self.encode(x)
        d = self.decode(e)
        return e, d

    def encode(self, x):
        e = F.sigmoid(self.encoder_fc1(x))
        e = F.sigmoid(self.encoder_fc2(e))
        return e

    def decode(self, x):
        d = F.sigmoid(self.decoder_fc1(x))
        d = F.sigmoid(self.decoder_fc2(d))
        return d
    
    

class autoencoder_cifar10(nn.Module):
    def __init__(self, code_dim=32):
        super(autoencoder_cifar10, self).__init__()
        self.code_dim = code_dim
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
			#nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            #nn.ReLU(),
 			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
            nn.ReLU(),
            #nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
            #nn.ReLU(),
        )
        #self.encoder_linear = nn.Linear(48 * 4 * 4, self.code_dim)
        #self.decoder_linear = nn.Linear(self.code_dim, 48 * 4 * 4)
        self.decoder_cnn = nn.Sequential(
            #nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            #nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
            nn.ReLU(),
			#nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            #nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
            #nn.Tanh()
        )
        
    def encoder(self, images):
        code = self.encoder_cnn(images)
        code = code.view(images.size(0), -1)
        #code = self.encoder_linear(code)
        code = nn.Sigmoid()(code)
        #code = nn.Tanh()(code)
        return code
    
    def decoder(self, code):
        out = code.view(code.size(0), 48, 4, 4)
        #out = nn.ReLU()(self.decoder_linear(code))
        #out = out.view(code.size(0), 48, 4, 4)
        out = self.decoder_cnn(out)
        return out
        

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded