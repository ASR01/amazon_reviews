import b_data_prep as b
from b_data_prep import TextData, ft_vectorizer, ft_vec
import c_model as c
from tqdm import tqdm
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import dill as pkl
import torch.nn.functional as f
from torchtext.vocab import FastText
#from torchsummary import summary

#import tqdm



size = 'test' #size of the set
# start_with_saved_data = False
max_seq_len = 32 #max length of the tokens list
batch_train_size = 16 # Dataloader batch size
batch_test_size = 16 # Dataloader batch size
emb_dim = 300 #embedded dimension
num_classes = 5
hidden_layer = 32

model = c.Classifier(max_seq_len, emb_dim, 64, 16,5) # from b_model.py
#summary(c.Classifier2, (16,32))


loss_function = nn.NLLLoss()

#loss_function = nn.CrossEntropyLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.parameters(), lr=0.005)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs = 20


train_df, test_df = b.prepare_data(size)
print(train_df.head)
dataset_te = b.TextData(test_df,  ft_vec, max_seq_len=32)
dataset_tr = b.TextData(train_df, ft_vec, max_seq_len=32)


# #Sequence legth for training
# if start_with_saved_data == False:

#     train_df, test_df = b.prepare_data(size)
#     print(train_df.shape, test_df.shape)

#     print('data_prepared', ' train: ', train_df.shape, 'test: ', test_df.shape)

#     dataset_te = b.TextData(test_df, max_seq_len=32)
#     print ('test_df ready')
#     file_te = open('./backup/dataset_te_c_tiny.pkl', 'wb') 
#     pkl.dump(dataset_te, file_te)
    
#     dataset_tr = b.TextData(train_df, max_seq_len=32)
#     print ('train_df ready')3

#     file_tr = open('./backup/dataset_tr_c_tiny.pkl', 'wb') 
#     pkl.dump(dataset_tr, file_tr)
    
# else:
    
#     print('Loading Data...')
#     with open('./backup/dataset_te_g_tiny.pkl', "rb") as file:
#         dataset_te = pkl.load(file)
#     with open('./backup/dataset_tr_g_tiny.pkl', "rb") as file:
#         dataset_tr = pkl.load(file)


# def ft_vectorizer(x):
#     vec = FastText("simple")
#     vec.vectors[1] = -torch.ones(vec.vectors[1].shape[0]) # replacing the vector associated with 1 (padded value) to become a vector of -1.
#     vec.vectors[0] = torch.zeros(vec.vectors[0].shape[0]) # replacing the vector associated with 0 (unknown) to become zeros
#     return vec.vectors[x]

# print(ft_vectorizer(468))
# print(ft_vectorizer(1))

print(dataset_tr[0][0])

def ft_collate(batch):
	# print('batch', type(batch))
	# print(batch)
	inputs = torch.stack([torch.stack([ft_vectorizer(token) for token in sentence[0]]) for sentence in batch])
	target = torch.LongTensor([item[1] for item in batch])

	return inputs, target


train_loader =  b.DataLoader(dataset_tr, batch_size=batch_train_size, collate_fn=ft_collate)
test_loader  =  b.DataLoader(dataset_te, batch_size=batch_test_size , collate_fn=ft_collate)
print(next(iter(train_loader)))

#Training
print_every = 10
epochs = 20

for e in tqdm(range(epochs)):
    running_loss = 0
    n_samples = 0
    n_correct = 0
    print(f"Epoch: {e+1}/{epochs}")
    model.train()
    for i, (sentences, labels) in enumerate(train_loader):
        
        sentences.resize_(sentences.size()[0], max_seq_len* emb_dim)
        
        #Forward
        output = model(sentences)   # 1) Forward pass
        loss = loss_function(output, labels) # 2) Compute loss
        #Backward
        optimizer.zero_grad()
        loss.backward()                  # 3) Backward pass
        optimizer.step()                 # 4) Update model
               
        running_loss += loss.item()
        
        
        
    print(f"\tIteration: {i}\t Loss: {running_loss:.4f}")
            
    if e % 2 == 0 and e > 0:
        file = './backup/model_last_' + str(device) + '.pkl'        
        model_file = open(file, 'wb') 
        pkl.dump(model, model_file)
        
        
    if e % 10 == 0 and e > 0:
        print('Evaluating')
        model.eval()
        for i, (sentences, labels) in enumerate(iter(test_loader)):
        
            with torch.no_grad():
                sentences.resize_(sentences.size()[0], 32* emb_dim)
                optimizer.zero_grad()
                     
                output = model.forward(sentences)   # 1) Forward pass
                
                ps = f.softmax(output, dim = 1)
                y_test = labels.numpy()    
                _, y_pred = torch.max(ps, dim = 1)
                n_samples += labels.size(0)
                n_correct += (y_pred == labels).sum().item()
            
            
        accuracy = n_correct/n_samples
        print('Accuracy of the model is: ', accuracy)
    
    
    
    



