import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from transformers import BertModel, BertTokenizer, get_scheduler

from fake_news import config
from fake_news.architecture import CustomBERTModel
from fake_news.make_data import make_data
from fake_news.predict import predict


# time based filename
def time_based_filename():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S")


# define format_time function
def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


logger.add(
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level:<8}</level>| <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    sink=Path(config.logs, f"log - {time_based_filename()}.txt"),
)


# define flat_accuracy function
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = np.argmax(labels, axis=1).flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def train(sentences, labels, lower=False):
    # Get pytorch device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set up tensorboard
    tbwriter = SummaryWriter(config.tensorboard)

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
            max_length=config.MAX_LENGTH,
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

    # split the dataset into train and validation
    train_size = int(config.TRAIN_SIZE * len(dataset))
    val_size = len(dataset) - train_size

    logger.info(f"Train size: {train_size}")
    logger.info(f"Validation size: {val_size}")

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # set train and validation dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=config.BATCH_SIZE,
    )

    validation_dataloader = DataLoader(
        val_dataset,
        sampler=RandomSampler(val_dataset),
        batch_size=config.BATCH_SIZE,
    )

    # define model
    model = CustomBERTModel(BertModel, BERT_MODEL=config.model_name).to(device)
    # set adamw optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    # set loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # set number of training steps
    total_steps = len(train_dataloader) * config.EPOCHS

    # set scheduler
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )

    training_stats = {}
    # start training clock
    start_time = time.time()

    # train model
    for epoch in range(config.EPOCHS):
        logger.info(f"Epoch {epoch + 1}/{config.EPOCHS}")
        logger.info("-" * 10)

        model.train()

        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            input_ids = batch[0].to(device)
            labels = batch[1].to(device)

            optimizer.zero_grad()

            logits = model(input_ids)

            loss = loss_fn(logits, labels)

            total_loss += loss.item()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()
            model.zero_grad()

        avg_train_loss = total_loss / len(train_dataloader)

        logger.info("  Average training loss: {0:.5f}".format(avg_train_loss))
        tbwriter.add_scalar(f"Training loss", avg_train_loss, epoch)

        # save model
        torch.save(model.state_dict(), Path(config.model_path, f"bert_{epoch}.pt"))

        # # evaluate model
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        predictions, true_labels = [], []

        with torch.no_grad():
            for batch in validation_dataloader:
                input_ids = batch[0].to(device)
                labels = batch[1].to(device)
                logits = model(input_ids)

                loss = loss_fn(logits, labels)

                total_eval_loss += loss.item()

                logits = logits.detach().cpu().numpy()
                label_ids = labels.to("cpu").numpy()

                total_eval_accuracy += flat_accuracy(logits, label_ids)

                predictions.extend(np.argmax(logits, axis=1).flatten().tolist())
                true_labels.extend(np.argmax(label_ids, axis=1).flatten().tolist())

            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            avg_val_loss = total_eval_loss / len(validation_dataloader)

            f1 = f1_score(true_labels, predictions)
            accuracy = accuracy_score(true_labels, predictions)

            training_stats[epoch] = {
                "Training Loss Per Epoch": total_loss,
                "Average Training Loss": avg_train_loss,
                "Validation Loss Per Epoch": total_eval_loss,
                "Average Validation Loss PE": avg_val_loss,
                "Average Validation Accuracy PE": avg_val_accuracy,
                "Training Time": format_time(time.time() - start_time),
                "F1 Score": f1,
                "Confusion Matrix": confusion_matrix(true_labels, predictions),
                "Accuracy (Sklearn)": accuracy,
            }

            tbwriter.add_scalar(f"Validation Accuracy", avg_val_accuracy, epoch)

            tbwriter.add_scalar(f"Training Loss Per Epoch", total_loss, epoch)
            tbwriter.add_scalar(f"Validation Loss Per Epoch", total_eval_loss, epoch)

            tbwriter.add_scalar(f"F1 Score", f1, epoch)

            tbwriter.add_scalar(f"Accuracy", accuracy, epoch)

    # save training stat in a txt file
    with open(Path("./fake_news/training_stats.txt"), "a") as f:
        f.write(str(training_stats) + "\n")

    print("Training complete!")


# python main function
if __name__ == "__main__":
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.derterministic = True
    torch.cuda.manual_seed_all(config.seed)

    # Log as making data
    logger.info("Making data...")

    # load data
    data = make_data(config.fakepath, config.truepath, config.savepath)

    #  get sentences and labels
    sentences = data["X"].values[: config.TOTAL_ROWS]
    labels = data["label"].values[: config.TOTAL_ROWS]

    # train model
    train(sentences, labels)
