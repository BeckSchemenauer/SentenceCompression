import os
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import BartTokenizer, BartForConditionalGeneration, DataCollatorForSeq2Seq
from transformers import Trainer, TrainingArguments
import pandas as pd

train_data = pd.read_csv("Release/compressionhistory.tsv", sep='\t', on_bad_lines='warn')

train_data["Source"] = train_data["Source"].astype(str)
train_data["Shortening"] = train_data["Shortening"].astype(str)
dic = {}
for i, sent in enumerate(train_data["Source"]):
    if sent in dic:
        list = dic[sent]
        list.append(train_data["Shortening"][i])
    else:
        dic[sent] = [train_data["Shortening"][i]]

for i in dic.keys():
    list = dic[i]
    dic[i] = sorted(list, key=len)

train_data["NewSource"] = None
train_data["NewShortening"] = None
for i, sent in enumerate(dic.keys()):
    train_data.loc[i, "NewSource"] = sent
    train_data.loc[i, "NewShortening"] = dic[sent][0]

train_data.dropna(inplace=True)
train_data["NewSource"] = train_data["NewSource"].astype(str)
train_data["NewShortening"] = train_data["NewShortening"].astype(str)
train_data.drop(["Source", "Shortening"], axis=1, inplace=True)
train_data = train_data[["NewSource", "NewShortening"]]


#print(train_data.head())