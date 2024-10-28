from collections.abc import Callable
from functools import partial
from itertools import chain
from pathlib import Path

import torch
from accelerate import Accelerator
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    DataCollatorForTokenClassification,
    PreTrainedTokenizerFast,
)

from ner.model import MODEL_NAME, NERModel, load_tokenizer, save_ner_model
from ner.utils import align_words, load_wnut16_dataset, unalign_tokens


def train_ner_model(
    model: NERModel,
    *,
    tokenizer: PreTrainedTokenizerFast,
    dataset_train: Dataset,
    dataset_validate: Dataset,
    metric_validate_cb: Callable[[list[list[int]]], tuple[float, float, float]],
    lr: float,
    num_epochs: int,
    batch_size: int,
) -> None:
    config = model.config

    accelerator = Accelerator()

    data_collator = DataCollatorForTokenClassification(tokenizer)

    train_dataloader = DataLoader(
        dataset_train,  # type: ignore
        collate_fn=data_collator,
        shuffle=True,
        batch_size=batch_size,
        num_workers=1,
    )

    validate_dataloader = DataLoader(
        dataset_validate,  # type: ignore
        collate_fn=data_collator,
        shuffle=False,
        batch_size=256,
        num_workers=1,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    model, optimizer, train_dataloader, validate_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, validate_dataloader
    )

    for epoch in range(1, num_epochs + 1):
        model.train()
        acc_loss = 0

        desc = f"Train {epoch}/{num_epochs}"
        progress_bar = tqdm(train_dataloader, desc)

        for data in progress_bar:
            optimizer.zero_grad()

            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]
            batch_token_ids = data["labels"]

            model_outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            logits = model_outputs["logits"].view(-1, config.num_labels)
            batch_token_ids = batch_token_ids.view(-1)

            loss = criterion(logits, batch_token_ids)

            accelerator.backward(loss)
            optimizer.step()

            loss = loss.item()
            acc_loss += loss

            progress_bar.set_postfix(loss=format(loss, ".4f"))

        avg_loss = acc_loss / len(train_dataloader)

        print(f"Train {epoch}/{num_epochs}: loss={avg_loss:.4f}")

        model.eval()

        col_token_ids = []

        desc = f"Validate {epoch}/{num_epochs}"
        progress_bar = tqdm(validate_dataloader, desc)

        for data in progress_bar:
            input_ids = data["input_ids"]
            attention_mask = data["attention_mask"]

            with torch.no_grad():
                model_outputs = model(
                    input_ids=input_ids, attention_mask=attention_mask
                )

            logits = model_outputs["logits"]
            col_token_ids += logits.argmax(-1).tolist()

        p, r, f = metric_validate_cb(col_token_ids)

        print(
            f"Validate {epoch}/{num_epochs}: percision={p:.4f}, recall={r:.4f}, f1-score={f:.4f}"
        )


def load_tokenized_dataset(
    dataset: dict[str, list[list[str]]],
    *,
    tokenizer: PreTrainedTokenizerFast,
    label2id: dict[str, int],
) -> Dataset:
    col_words = dataset["words"]
    col_labels = dataset["labels"]

    tokenized_inputs = tokenizer(
        col_words,
        truncation=True,
        is_split_into_words=True,
    )

    col_word_ids = [[label2id[tag] for tag in tags] for tags in col_labels]

    col_word_indices = [tokenized_inputs.word_ids(i) for i, _ in enumerate(col_words)]

    col_token_ids = [*map(align_words, col_word_ids, col_word_indices)]

    return Dataset.from_dict(
        {
            **tokenized_inputs,
            "labels": col_token_ids,  # huggingface forces the column name to be "labels"
            "word_ids": col_word_ids,
            "word_indices": col_word_indices,
        }
    )


def metric_callback(
    col_pred_token_ids: list[list[int]],
    *,
    col_word_ids: list[list[int]],
    col_word_indices: list[list[int | None]],
) -> tuple[float, float, float]:
    flat_true_word_ids = [*chain.from_iterable(col_word_ids)]
    flat_pred_word_ids = [
        *chain.from_iterable(map(unalign_tokens, col_pred_token_ids, col_word_indices))
    ]
    p, r, f, s = precision_recall_fscore_support(
        flat_true_word_ids,
        flat_pred_word_ids,
        average="micro",
    )
    return float(p), float(r), float(f)


def main(*, lr: float, num_epochs: int, batch_size: int) -> None:
    res_path = Path("resources")

    dataset_train = load_wnut16_dataset(
        res_path / "wnut_16" / "train.txt", with_labels=True
    )
    dataset_validate = load_wnut16_dataset(
        res_path / "wnut_16" / "dev.txt", with_labels=True
    )

    labels_set = set(chain.from_iterable(dataset_train["labels"]))

    id2tag = dict(enumerate(sorted(labels_set)))
    tag2id = {tag: i for i, tag in id2tag.items()}

    tokenizer = load_tokenizer()

    tokenized_dataset_train = load_tokenized_dataset(
        dataset_train,
        tokenizer=tokenizer,
        label2id=tag2id,
    )
    tokenized_dataset_validate = load_tokenized_dataset(
        dataset_validate,
        tokenizer=tokenizer,
        label2id=tag2id,
    )

    config = AutoConfig.from_pretrained(
        MODEL_NAME,
        label2id=tag2id,
        id2label=id2tag,
        num_labels=len(labels_set),
    )
    model = NERModel(config, pretrained=True)
    # fix the feature extraction module
    model.feature_extraction.requires_grad_(False)

    metric_validate_cb = partial(
        metric_callback,
        col_word_ids=tokenized_dataset_validate["word_ids"],
        col_word_indices=tokenized_dataset_validate["word_indices"],
    )

    tokenized_dataset_train = tokenized_dataset_train.remove_columns(
        ["word_ids", "word_indices"]
    )
    tokenized_dataset_validate = tokenized_dataset_validate.remove_columns(
        ["word_ids", "word_indices", "labels"]
    )

    train_ner_model(
        model,
        tokenizer=tokenizer,
        dataset_train=tokenized_dataset_train,
        dataset_validate=tokenized_dataset_validate,
        metric_validate_cb=metric_validate_cb,
        lr=lr,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )

    save_ner_model(model, res_path / "ner_model.pt")


if __name__ == "__main__":
    main(lr=1e-5, num_epochs=10, batch_size=8)
