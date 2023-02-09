import time
import numpy as np
import torch

from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import DataLoader, RandomSampler

from transformers import BertModel, BertTokenizer
from fake_news.architecture import CustomBERTModel
from fake_news import config

from fake_news.make_data import make_data
from fake_news.predict import predict


from loguru import logger

logger.add(
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level:<8}</level>| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    sink="log.txt",
)


# define flat_accuracy function
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = np.argmax(labels, axis=1).flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# define format_time function
def format_time(elapsed):
    import datetime

    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def train(sentences, labels, lower=False):
    # Get pytorch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_ids = []
    targets = []

    # define bert tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    for i in range(len(sentences)):
        sentences[i] = sentences[i].strip()

        if lower:
            sentences[i] = sentences[i].lower()

        inputs = tokenizer.encode_plus(
            sentences[i],
            add_special_tokens=True,
            truncation=True,
            padding="max_length",
            max_length=64,
            return_tensors="pt",
        ).to(device)

        input_ids.append(inputs["input_ids"].squeeze(0))

        # convert labels to one-hot encoding
        target = torch.zeros(2)
        target[labels[i]] = 1
        targets.append(target)

    # convert to tensors
    input_ids = torch.stack(input_ids, dim=0)
    targets = torch.stack(targets, dim=0)

    # create dataset
    dataset = TensorDataset(input_ids, targets)
    # create dataloader
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # define model
    model = CustomBERTModel(BertModel, BERT_MODEL="bert-base-uncased").to(device)

    # set adamw optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    # set loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # set number of training steps
    total_steps = len(dataloader) * config.EPOCHS
    # set scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-5, steps_per_epoch=len(dataloader), epochs=config.EPOCHS
    )

    # start training clock
    start_time = time.time()

    # train model
    for epoch in range(config.EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{config.EPOCHS}")
        logger.info("-" * 10)

        model.train()

        total_loss = 0

        for step, batch in enumerate(dataloader):
            input_ids = batch[0].to(device)
            labels = batch[1].to(device)

            logits = model(input_ids)

            loss = loss_fn(logits, labels)

            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            model.zero_grad()

        avg_train_loss = total_loss / len(dataloader)
        logger.info("  Average training loss: {0:.2f}".format(avg_train_loss))

        # save model
        torch.save(model.state_dict(), f"./fake_news/models/bert_{epoch}.pt")

        # # evaluate model
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        for batch in dataloader:
            input_ids = batch[0].to(device)
            labels = batch[1].to(device)
            logits = model(input_ids)

            loss = loss_fn(logits, labels)

            total_eval_loss += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = labels.to("cpu").numpy()

            total_eval_accuracy += flat_accuracy(logits, label_ids)

        avg_val_accuracy = total_eval_accuracy / len(dataloader)
        logger.info("  Accuracy: {0:.2f}".format(avg_val_accuracy))

        avg_val_loss = total_eval_loss / len(dataloader)
        logger.info("  Validation Loss: {0:.2f}".format(avg_val_loss))

        logger.info(
            "  Training epcoh took: {:}".format(format_time(time.time() - start_time))
        )


# python main function
if __name__ == "__main__":
    # Log as making data
    logger.info("Making data...")

    # load data
    data = make_data(config.fakepath, config.truepath, config.savepath)

    #  get sentences and labels
    sentences = data["title"].values[:1000]
    labels = data["label"].values[:1000]

    # train model
    train(sentences, labels)
