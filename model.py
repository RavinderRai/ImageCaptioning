import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first = True, dropout = 0.5)
        
        #self.dropout = nn.Dropout(0.5)
        
        self.linear = nn.Linear(hidden_size, vocab_size)
        
        

    
    def forward(self, features, captions):
        
        x = self.embedding(captions[:, :-1])
        
        lstm_input = torch.cat((features.unsqueeze(dim = 1), x), dim=1)
        
        x, hidden = self.lstm(lstm_input)
        
        #x = self.dropout(x)
        
        x = self.linear(x)
        
        return x

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
            
        
        cap = []
        hidden = (torch.randn(self.num_layers, 1, self.hidden_size).to(inputs.device), torch.randn(1, 1, 512).to(inputs.device))
        
        for i in range(max_len):
            
            lstm_out, hidden = self.lstm(inputs, hidden)
            
            
            outputs = self.linear(lstm_out)
            
            
            outputs = outputs.squeeze(1)
            
            #outputs[0][0]=0
            
            word  = outputs.argmax(dim=1)
            
            
            #print(outputs.max(dim = 1))
            #print(outputs.argmax(dim=1))
            
            
            #print(word.item())
            
            cap.append(word.item())
            
            
            #inputs = self.embedding(word.unsqueeze(0))
            
                                   
            inputs = self.embedding(word)
            inputs = inputs.unsqueeze(1) 
            
            
            #if i>6:
             #   inputs = self.embedding(word)
             #   inputs = inputs.unsqueeze(1)
           
            
        return cap
        
        
        