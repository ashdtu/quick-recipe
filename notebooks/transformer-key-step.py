#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
from transformers import AutoTokenizer


# In[59]:


data = pd.read_csv("../youcook2/reviewed_0812_pred.csv")


# In[60]:


# data["Sentence"].head()


# In[6]:


# len(data["VideoUrl"].unique())


# In[9]:


tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


# In[10]:


def tokenize_text(text):
    return tokenizer(text, truncation=True)


# In[18]:


# b = tokenize_text(data["Sentence"][0])


# In[19]:


import json


# In[20]:


with open("../youcook2/2.cooking_vocab_filtered_captions.tmp.json", "rb") as f:
    full_data = json.load(f)


# In[28]:


yt_id, split = zip(*[(x["youtube_id"], x["partition"]) for x in full_data])


# In[31]:


import numpy as np


# In[38]:


from collections import Counter


# In[42]:


# data["yt_id"] = data["VideoUrl"].apply(lambda x: x.split('watch?v=')[1])


# In[44]:


# data_yt_ids = data["yt_id"].unique()


# In[49]:


# len(yt_id), len(data_yt_ids)


# In[164]:


seed = 23456
np.random.seed(seed)


# In[53]:


# data.columns


# In[165]:


train_len = int(0.70*len(data))


# In[166]:


indices = list(range(len(data)))
np.random.shuffle(indices)


# In[171]:


all_text, all_labels = data["Sentence"].to_numpy(), data["IsUsefulSentence"].to_numpy()


# In[173]:


train_data, train_labels = all_text[indices[:train_len]], all_labels[indices[:train_len]]


# In[174]:


val_data, val_labels = all_text[indices[train_len:]], all_labels[indices[train_len:]]


# In[177]:


train_encodings = tokenizer(list(train_data), truncation=True, padding=True)
val_encodings = tokenizer(list(val_data), truncation=True,padding=True)


# In[178]:


import torch

class YouCookData(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# In[179]:


train_dataset = YouCookData(train_encodings, list(train_labels))
val_dataset = YouCookData(val_encodings, list(val_labels))


# In[88]:


from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# In[89]:

import evaluate

accuracy = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# In[87]:


import os
os.environ["WANDB_DISABLED"] = "true"


# In[94]:


id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1}


# In[95]:


from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
)


# In[180]:


training_args = TrainingArguments(
    output_dir="../models/",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="../logs/",
    load_best_model_at_end=True,
    logging_steps=50
    
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

