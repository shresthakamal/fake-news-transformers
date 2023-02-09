import numpy as np
import torch

from transformers import BertModel

# Load tokenizer and model


class CustomBERTModel(torch.nn.Module):
    def __init__(self, bert_model, BERT_MODEL="bert-base-uncased"):
        super(CustomBERTModel, self).__init__()

        self.bert_model = BertModel.from_pretrained(
            BERT_MODEL, output_hidden_states=True
        )

        # set a linear layer to map the hidden states to 64 dimensions
        self.linear = torch.nn.Linear(768, 64)
        # set another linear layer to map the 64 dimensions to 2 dimensions
        self.linear2 = torch.nn.Linear(64, 2)
        # set a dropout layer
        self.dropout = torch.nn.Dropout(0.1)
        # set a relu activation function
        self.relu = torch.nn.ReLU()

    def forward(self, input_ids):
        outputs = self.bert_model(input_ids)

        # pass the last hidden state of the token `[CLS]` to the linear layer
        x = self.linear(outputs[0][:, 0, :])

        # pass the output of the linear layer to the relu activation function
        x = self.relu(x)

        # pass the output of the relu activation function to the dropout layer
        x = self.dropout(x)

        # pass the output of the dropout layer to the second linear layer
        x = self.linear2(x)

        # set a softmax activation function
        x = torch.nn.functional.softmax(x, dim=1)

        return x
