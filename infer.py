from pathlib import Path

import torch
from accelerate import Accelerator
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    DataCollatorForTokenClassification,
    PreTrainedTokenizerFast,
)

from ner.model import NERModel, load_ner_model, load_tokenizer
from ner.utils import load_wnut16_dataset, unalign_tokens


def infer_ner_model(
    model: NERModel,
    *,
    tokenizer: PreTrainedTokenizerFast,
    dataset: Dataset,
) -> list[list[int]]:
    accelerator = Accelerator()

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    dataloader = DataLoader(
        dataset,  # type: ignore
        collate_fn=data_collator,
        batch_size=128,
        pin_memory=True,
        num_workers=1,
    )

    model, dataloader = accelerator.prepare(model, dataloader)

    model.eval()

    col_token_ids = []

    progress_bar = tqdm(dataloader, desc="Infer")

    for data in progress_bar:
        input_ids = data["input_ids"]
        attention_mask = data["attention_mask"]

        with torch.no_grad():
            model_outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        logits = model_outputs["logits"]
        col_token_ids += logits.argmax(-1).tolist()

    return col_token_ids


def main() -> None:
    res_path = Path("resources")
    model_path = res_path / "ner_model.pt"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = load_ner_model(res_path / "ner_model.pt")

    dataset = load_wnut16_dataset(res_path / "wnut_16" / "test.txt", with_labels=False)

    col_words = dataset["words"]

    dataset_size = len(col_words)

    tokenizer = load_tokenizer()

    tokenized_inputs = tokenizer(
        col_words,
        truncation=True,
        is_split_into_words=True,
    )

    tokenized_dataset = Dataset.from_dict({**tokenized_inputs})

    col_word_indices = [tokenized_inputs.word_ids(i) for i in range(dataset_size)]

    col_token_ids = infer_ner_model(
        model, tokenizer=tokenizer, dataset=tokenized_dataset
    )

    id2label = model.config.id2label

    col_word_ids = map(unalign_tokens, col_token_ids, col_word_indices)

    col_word_and_labels = [
        [(word, id2label[word_id]) for word, word_id in zip(words, word_ids)]
        for words, word_ids in zip(col_words, col_word_ids)
    ]

    formatted_word_tags = "\n\n".join(
        "\n".join(f"{word}\t{tag}" for word, tag in word_tag)
        for word_tag in col_word_and_labels
    )

    Path("results.txt").write_text(formatted_word_tags, encoding="utf-8")


if __name__ == "__main__":
    main()
