import streamlit as st
import torch
import torch.nn.functional as f
from b_data_prep import padding, encoder, preprocessing, ft_vec, ft_vectorizer 
from c_model import Classifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def bar_chart(ps):

    #Creating the dataset
    # Sentiment = ['Negative', 'Positive'] In case we use the negative/positive version
    Sentiment = [1,2,3,4,5]
    values = ps.data.numpy().squeeze()

    fig = plt.figure(figsize = (10, 5))
    plt.barh(Sentiment, values, color = ['black', 'red', 'orange', 'yellow', 'green'])

    plt.xlabel("Percentage")
    plt.ylabel("Sentiment")
    plt.title("Probability of stars with the given text")
    st.pyplot(fig)


st.title("Sentiment analysis based in the Amazon Reviews File.")


text = st.text_area('Just insert the text you want to check for rating and press Calculate', value="")

st.write('Please keep in mind that a longer review text means a most accurate value')

calculate = st.button('Calculate')

# filename = './backup/model_best_2_cuda.pt' In case we use the positive negative version
filename = './backup/model_best_5_cuda.pt'

 
max_seq_len = 32 
ft_dim = 300 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load(filename, map_location = device)
model.eval()

# 

sequence = padding(encoder(preprocessing(str(text)),ft_vec ), max_seq_len)
             
print(sequence)

input = ft_vectorizer(sequence)

input.resize_(1, max_seq_len* ft_dim)
# print(input.shape)

output = model.forward(input)
ps = f.softmax(output, dim = 1)
_, y_pred = torch.max(ps, dim = 1)

if calculate:
    bar_chart(ps)
	