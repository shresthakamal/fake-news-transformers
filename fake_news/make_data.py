import os
import re
import string
import pandas as pd
from sklearn.utils import shuffle
from fake_news import config


def remove_special_characters(text):
    text = text.lower()
    text = re.sub("\[.*?\]", "", text)
    text = re.sub("\\W", " ", text)
    text = re.sub("https?://\S+|www\.\S+", "", text)
    text = re.sub("<.*?>+", "", text)
    text = re.sub("[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub("\n", "", text)
    text = re.sub("\w*\d\w*", "", text)
    return text


def make_data(fakepath, truepath, savepath):
    if os.path.exists(config.savepath):
        data = pd.read_csv(config.savepath)
        return data

    fake = pd.read_csv(fakepath)
    fake["label"] = 0

    true = pd.read_csv(truepath)
    true["label"] = 1

    data = pd.concat([fake, true], ignore_index=True)

    data["text"] = data["text"].apply(remove_special_characters)
    data["title"] = data["title"].apply(remove_special_characters)

    data["X"] = data["title"] + "[SEP]" + data["text"]

    data = shuffle(data)

    data.to_csv(savepath, index=False)

    return data


# python main function
if __name__ == "__main__":
    fakepath = "./data/Fake.csv"
    truepath = "./data/True.csv"
    savepath = "./fake_news/data/data.csv"

    make_data(fakepath, truepath, savepath)
