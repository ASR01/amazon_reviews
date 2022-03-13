import torchtext
import pandas as pd

torchtext.datasets.AmazonReviewFull(root='./data/', split=("train", "test"))

df = pd.read_csv("./data/AmazonReviewFull/amazon_review_full_csv/test.csv", header=None)
df.to_csv('./data/test.csv', index = None)

df = pd.read_csv("./data/AmazonReviewFull/amazon_review_full_csv/train.csv", header=None)
df.to_csv('./data/train.csv', index = None)

train = 0.8
small = 1000
med = 50000
large = 150000
xlarge = 1000000

df = pd.read_csv("./data/test.csv", nrows=int(small*(1-train)), header=None)
df.to_csv('./data/test_small.csv', index = None)

df = pd.read_csv("./data/train.csv", nrows=int(small*train), header=None)
df.to_csv('./data/train_small.csv', index = None)


df = pd.read_csv("./data/test.csv", nrows=int(med*(1-train)), header=None)
df.to_csv('./data/test_med.csv', index = None)

df = pd.read_csv("./data/train.csv", nrows=int(med*train), header=None)
df.to_csv('./data/train_med.csv', index = None)

df = pd.read_csv("./data/test.csv", nrows=int(large*(1-train)), header=None)
df.to_csv('./data/test_large.csv', index = None)

df = pd.read_csv("./data/train.csv", nrows=int(large*train), header=None)
df.to_csv('./data/train_large.csv', index = None)

df = pd.read_csv("./data/test.csv", nrows=int(xlarge*(1-train)), header=None)
df.to_csv('./data/test_large.csv', index = None)

df = pd.read_csv("./data/train.csv", nrows=int(xlarge*train), header=None)
df.to_csv('./data/train_large.csv', index = None)



