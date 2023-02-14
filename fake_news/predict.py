import torch
import numpy as np
from fake_news import config

from transformers import BertModel, BertTokenizer
from fake_news.architecture import CustomBERTModel
from pathlib import Path
from loguru import logger


# define predict function
def predict(sentence, lower=False):
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
        max_length=config.MAX_LENGTH,
        return_tensors="pt",
    ).to(device)

    input_ids = inputs["input_ids"]

    # define model
    model = CustomBERTModel(BertModel, BERT_MODEL="bert-base-uncased").to(device)

    # load model
    model.load_state_dict(torch.load(Path(config.model_path, "bert_9.pt")))

    # set model to evaluation mode
    model.eval()

    logits = model(input_ids)

    logits = logits.detach().cpu().numpy()

    return np.argmax(logits, axis=1)


# Python main function
if __name__ == "__main__":
    # predict
    pred = predict(
        """Queensland result leaves Australian PM closer to edge,"SYDNEY (Reuters) - The loss of a state election in Queensland has stepped up pressure on Australian Prime Minister Malcolm Turnbull, who risks losing control of parliament at a by-election next month. Three Australian prime ministers have been ousted by their own parties since 2010, and a splintering of the conservative base in Queensland has raised questions over how long Turnbull s premiership can survive. Opinion polls already show his popularity at a record low.  Queensland s Liberal National Party (LNP), which replicates the federal coalition made up of Turnbull s Liberal Party and its partner the National Party, was hurt by voters, particularly in regional and rural areas, defecting to Pauline Hanson s right-wing, populist One Nation party. Vote counting is still underway, but the conservative divide has left the Labor Party on track to form the government in the coal-rich northeastern state. Smarting from this latest setback, Turnbull reminded voters on Monday that if they backed One Nation at the next federal election it could play into the hands of the center opposition.  Everyone is entitled to cast their vote as they see fit but the voting for One Nation in the Queensland election has only assisted the Labor Party,   Turnbull told reporters in the city of Wollongong, south of Sydney. The next federal election is due either in late 2018 or early 2019. But first up is the Bennelong by-election on Dec. 16. Should the Liberals lose the seat in Sydney s north, Turnbull would have to negotiate with independents and small parties to retain control of the House of Representatives, where the government is formed. It could heighten chances of deadlock between the two houses of parliament, which might force Turnbull to call an early election, just as he did last year. Regarded as a moderate, Turnbull has trouble holding on to voters leaking to the right following the resurgence of Hanson s  anti-immigration party, according to Queensland University of Technology political science expert Clive Bean.  In recent times Queensland has often been one of the states that has made the difference when it comes to whether the coalition wins government or not,  said Bean.  The seats that tend to bleed votes to One Nation do tend to be seats where the LNP is traditionally stronger.  Forecast to win just one seat in Queensland, One Nation polled almost 14 percent of the vote, spoiling the LNP s chances of taking the state off Labor. At the federal level, the ruling coalition s fragility has been exacerbated by rules forcing lawmakers holding dual nationality, which is prohibited, to recontest seats. Bennelong is one such seat, and should defeat there lead to the coalition losing control over the House of Representatives it would immediately undermine the prime minister s efforts to  stave off an inquiry into Australia s scandal-hit major banks. While Turnbull has distanced himself from the Queensland election result, maverick coalition member George Christensen tweeted an apology to voters who switched allegiance to One Nation, blaming the federal government for not standing up for conservative values.  A lot of that rests with the Turnbull govt, it s (sic) leadership & policy direction,  the tweet said. ",worldnews,"November 27, 2017 """,
        lower=True,
    )

    # map prediction to label
    label = "FAKE" if pred == 0 else "REAL"

    logger.info(label)
