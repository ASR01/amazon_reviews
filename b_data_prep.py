from collections import Counter
import numpy as np
import pandas as pd
import spacy
import torch
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import FastText
from tqdm import tqdm, tqdm_notebook

nlp = spacy.load("en_core_web_sm")

## Define out fasttext vectorizer 
ft_vec = FastText("simple")
ft_vec.vectors[1] = -torch.ones(ft_vec.vectors[1].shape[0]) 
ft_vec.vectors[0] = torch.zeros(ft_vec.vectors[0].shape[0])


'''
tiny = 100
small = 10000
med = 100000
large = 500000
all = fulllength
'''


def loading_data(size):
	
	if size == 'all':
		df_train = pd.read_csv('./data/train.csv', header=None)
		df_test  = pd.read_csv('./data/test.csv', header=None)

	else:
		df_train = pd.read_csv('./data/train_'+ str(size) + '.csv', header=None)
		df_test  = pd.read_csv('./data/test_' + str(size) + '.csv', header=None)

	return (df_train, df_test)


def adjust_df(df):

	df.rename({0:"star", 1:"rating1", 2:"rating2"}, axis=1, inplace=True)
	df["review"] = df["rating1"] + " " +  df["rating2"]
	df.drop(columns=["rating1", "rating2"], inplace=True)
	df.star = df.star.apply(lambda x: int(x) -1 if x >0 else 0) # to keep the stars
	# df.star = df.star.apply(lambda x: 1 if x >3 else 0) # to have just positiv and negative
	df['leng'] = df.review.apply(lambda x: len(str(x).split()))	
	df.drop(df[df['leng'] < 10].index, inplace = True)
	df.reset_index(drop=True, inplace = True)
	return df

## To be used in the AMZ_Dataset 
##################################################################
def preprocessing(sentence):
	'''
	string containing the sentence we want to preprocess return the tokens list
	'''
	doc = nlp(sentence)
	tokens = [token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
	#print(tokens)
	return tokens

def token_encoder(token, vec):
	if token == "<pad>":
		return 1 # the padding vector is set to -1
	else:
		try:
			return vec.stoi[token]
		except:
			return 0

def encoder(tokens, vec):
	return [token_encoder(token, vec) for token in tokens]


def padding(list_of_indexes, max_seq_len, padding_index=1):
	output = list_of_indexes + (max_seq_len - len(list_of_indexes))*[padding_index]
	return output[:max_seq_len]

def ft_vectorizer(x):
    
    return ft_vec.vectors[x]

##################################################

class TextData(Dataset):
	def __init__(self, passed_df, ft_vect, max_seq_len=32): 
		
		self.max_seq_len = max_seq_len
		self.labels = passed_df.star

		self.sequences = [padding(encoder(preprocessing(str(sequence)),ft_vect ), max_seq_len) for sequence in passed_df.review.tolist()]


	def __len__(self):
		return len(self.sequences)
	
	def __getitem__(self, idx):
		return self.sequences[idx], self.labels[idx]



def prepare_data(size):
	df_train, df_test = loading_data(size)
	# print('load ready')
	df_train = adjust_df(df_train)
	df_train.to_csv('./data/train_changed.csv')
	# print('adjust train ready')
	df_test = adjust_df(df_test)
	df_test.to_csv('./data/train_changed.csv')
	# print('adjust test ready')
	return df_train, df_test

# def tensor2csv(tensor):
# 	t_np = tensor.numpy() #convert to Numpy array
# 	df = pd.DataFrame(t_np) #convert to a dataframe
# 	df.to_csv("./tensor0.csv",index=False) #save to file



def main():
#	loading_data('small')
	# print(ft_vectorizer(468))
	pass

if __name__ == '__main__':
	main()


