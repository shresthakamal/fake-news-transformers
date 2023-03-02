import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
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

    """
    The optimizer.zero_grad() function resets the gradients of all the trainable parameters in the model that are managed by the optimizer.
    This is typically used inside the training loop, before calling loss.backward() to compute the gradients.

    The model.zero_grad() function resets the gradients of all the trainable parameters in the model that are not managed by the optimizer.
    This is typically used outside the training loop, before calling loss.backward() to compute the gradients when not using an optimizer.
    """

    # set model to training mode
    model.train()

    # clear gradients
    model.zero_grad()

    # train model
    for epoch in range(config.EPOCHS):
        # log progress
        logger.info(f"Epoch {epoch + 1}/{config.EPOCHS}")

        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            input_ids = batch[0].to(device)
            labels = batch[1].to(device)

            logits = model(input_ids)

            loss = loss_fn(logits, labels)

            total_loss += loss.item()

            # clear gradients
            # This is called just before loss.backward() to clear the gradients of all optimized torch.Tensors.
            optimizer.zero_grad()

            # Perform back propagation to calculate the gradients
            loss.backward()

            # this is to clip the norm of the gradients to 1.0
            # This prevents the "exploding gradients" problem
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Now use those calculated gradients to update the parameters of our model
            # The optimizer dictates the "update rule" - how the parameters are modified based on their gradients, the learning rate, etc.
            # The learning rate is a very important hyperparameter that controls how much we are adjusting the parameters of our model with respect to the loss gradient.
            optimizer.step()

            # Update the learning rate
            scheduler.step()

        avg_train_loss = total_loss / len(train_dataloader)

        # save model
        torch.save(model.state_dict(), Path(config.model_path, f"bert_{epoch}.pt"))

        # # evaluate model
        # Put the model in evaluation mode--the dropout layers behave differently during evaluation.
        model.eval()

        total_eval_accuracy = 0
        total_eval_loss = 0
        predictions, true_labels = [], []

        # Telling the model not to compute or store gradients, saving memory and speeding up validation
        # https://pytorch.org/docs/stable/generated/torch.no_grad.html
        # sets all requires_grad flags to false

        with torch.no_grad():
            for batch in validation_dataloader:
                input_ids = batch[0].to(device)
                labels = batch[1].to(device)

                logits = model(input_ids)

                loss = loss_fn(logits, labels)

                # move logits and labels to cpu to calculate metrics and convert to numpy
                logits = logits.detach().cpu().numpy()
                label_ids = labels.to("cpu").numpy()

                total_eval_loss += loss.item()
                total_eval_accuracy += flat_accuracy(logits, label_ids)

                predictions.extend(np.argmax(logits, axis=1).flatten().tolist())
                true_labels.extend(np.argmax(label_ids, axis=1).flatten().tolist())

            avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
            avg_val_loss = total_eval_loss / len(validation_dataloader)

            # Metrics from Sklearn
            f1 = f1_score(true_labels, predictions)
            accuracy = accuracy_score(true_labels, predictions)
            cm = confusion_matrix(true_labels, predictions)

            training_stats[epoch] = {
                "Training Loss Per Epoch": total_loss,
                "Average Training Loss": avg_train_loss,
                "Validation Loss Per Epoch": total_eval_loss,
                "Average Validation Loss PE": avg_val_loss,
                "Average Validation Accuracy PE": avg_val_accuracy,
                "Training Time": format_time(time.time() - start_time),
                "F1 Score": f1,
                "Confusion Matrix": cm,
                "Accuracy (Sklearn)": accuracy,
            }

            # plot training loss per epoch and validation loss per epoch in same tensorboard graph
            tbwriter.add_scalars(
                "Loss",
                {
                    "Training Loss Per Epoch": total_loss,
                    "Validation Loss Per Epoch": total_eval_loss,
                },
                epoch,
            )

            # plot validation accuracy per epoch and sklearn accuracy per epoch in same tensorboard graph
            tbwriter.add_scalars(
                "Accuracy",
                {
                    "Validation Accuracy Per Epoch": avg_val_accuracy,
                    "Accuracy (Sklearn)": accuracy,
                },
                epoch,
            )

            tbwriter.add_scalar(f"F1 Score", f1, epoch)

    # create a pandas dataframe from training stats
    df_stats = pd.DataFrame(data=training_stats).transpose()

    # save dataframe to csv
    df_stats.to_csv(Path("./fake_news/training_stats.csv"))

    print(df_stats)

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
