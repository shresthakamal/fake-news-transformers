import torch
import numpy as np
import pandas as pd
from fake_news import config
from tqdm import tqdm
from transformers import BertModel, BertTokenizer
from fake_news.architecture import CustomBERTModel
from pathlib import Path


# define predict function
def predict(sentence, lower, device, model, tokenizer ):
    

    if lower:
        sentence = sentence.lower()

    inputs = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=config.MAX_LENGTH,
        return_tensors="pt",
    ).to(device)

    input_ids = inputs["input_ids"]

    # define model
    

    # load model
    model.load_state_dict(torch.load(Path(config.model_path, "bert_9.pt")))

    # set model to evaluation mode
    model.eval()

    logits = model(input_ids)

    logits = logits.detach().cpu().numpy()

    return np.argmax(logits, axis=1)


# Python main function
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = CustomBERTModel(BertModel, BERT_MODEL="bert-base-uncased").to(device)

    test = pd.read_csv("./data/test.csv")
    test = test.fillna(" ")

    test["X"] = test["author"] + "[SEP]" + test["title"] + "[SEP]" + test["text"]

    submit = []

    for index, sentence in tqdm(enumerate(test["X"])):

        pred = predict(sentence, lower=False, device = device, model = model, tokenizer = tokenizer)
        print([test.iloc[index]["id"], pred[0]])
        submit.append([test.iloc[index]["id"], pred[0]])
   
    df = pd.DataFrame(submit, columns=["id", "label"])
    df.to_csv("submit.csv", index=False)

    
