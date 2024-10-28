from os import PathLike
from typing import Any, cast

import torch
from torch import nn
from transformers import (
    AutoModel,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizerFast,
)

MODEL_NAME = "dslim/distilbert-NER"


class NERModel(PreTrainedModel):
    def __init__(self, config: PretrainedConfig, *, pretrained: bool) -> None:
        super().__init__(config)
        if pretrained:
            feature_extraction = AutoModel.from_pretrained(
                config.name_or_path, config=config
            )
        else:
            feature_extraction = AutoModel.from_config(config)
        self.feature_extraction = cast(nn.Module, feature_extraction)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> dict[str, Any]:
        with torch.no_grad():
            output = self.feature_extraction(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        feature = self.dropout(output.last_hidden_state)
        logits = self.classifier(feature)
        return {"logits": logits}


def save_ner_model(model: NERModel, path: str | PathLike[str]) -> None:
    torch.save(
        {
            "config": model.config,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_ner_model(path: str | PathLike[str]) -> NERModel:
    checkpoint = torch.load(path, weights_only=False)
    model = NERModel(checkpoint["config"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    return model


def load_tokenizer() -> PreTrainedTokenizerFast:
    return cast(PreTrainedTokenizerFast, AutoTokenizer.from_pretrained(MODEL_NAME))
