from torch import nn
import torch.nn.functional as f




# class Classifier(nn.Module):
#     def __init__(self, max_seq_len, emb_dim, hidden1, hidden2, num_classes):
#         super(Classifier, self).__init__()
#         self.fc1 = nn.Linear(max_seq_len*emb_dim, hidden1)
#         self.fc2 = nn.Linear(hidden1, hidden2)
#         self.fc3 = nn.Linear(hidden2, num_classes)
#         # self.out = nn.Softmax(dim=1)
    
    
#     def forward(self, inputs):
        
#         x = f.relu(self.fc1(inputs.squeeze(1).float()))
#         x = f.relu(self.fc2(x))
#         return self.fc3(x)
        
        
     

class Classifier(nn.Module):
    def __init__(self, max_seq_len, emb_dim, hidden1, hidden2, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(max_seq_len*emb_dim, hidden1*32)
        self.fc2 = nn.Linear(hidden1*32, hidden1*8)
        self.fc3 = nn.Linear(hidden1*8, hidden1)
        self.fc4 = nn.Linear(hidden1, hidden2)
        self.fc5 = nn.Linear(hidden2, num_classes)
        # self.out = nn.Softmax(dim=1)
    
    
    def forward(self, inputs):
        x = f.relu(self.fc1(inputs.squeeze(1).float()))
        x = f.relu(self.fc2(x))
        x = f.relu(self.fc3(x))
        x = f.relu(self.fc4(x))
        return self.fc5(x)
        