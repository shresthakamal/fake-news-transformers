import torch

from transformers import BertModel, BertTokenizer

# define predict function
def predict(sentence, lower = False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define bert tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    if lower:
        sentence = sentence.lower()
    
    inputs = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        ).to(device)

    input_ids = inputs["input_ids"]

    # define model
    model = CustomBERTModel(BertModel, BERT_MODEL = "bert-base-uncased").to(device)

    # load model
    model.load_state_dict(torch.load("../models/bert_9.pt"))

    # set model to evaluation mode
    model.eval()

    logits = model(input_ids)

    logits = logits.detach().cpu().numpy()

    return np.argmax(logits, axis=1)


# Python main function  
if __name__ == "__main__":

    # predict
    print(predict(data["title"][0], lower = True))