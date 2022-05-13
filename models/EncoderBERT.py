from abc import ABCMeta, abstractmethod
import torch

from .utils import update_model_state_dict


class EncoderBERT(torch.nn.Module, metaclass=ABCMeta):
    r"""Fully-connected Network on top of BERT"""

    def __init__(
        self,
        encoder,
        num_labels=2,
        trainable_encoder=True,
    ):

        super(EncoderBERT, self).__init__()

        self.num_labels = num_labels
        self.encoder = encoder
        if not trainable_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

    @classmethod
    def from_pretrained(cls, saved_path="", load_strategy="best", **init_kwargs):
        model = cls(**init_kwargs)
        model = update_model_state_dict(model, saved_path, load_strategy)
        return model

    @abstractmethod
    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        r"""placeholder labels argument is provided for the compatibility with
        the huggingface dataset and Trainer api's compute_metrics."""
        pass

    def get_in_out_channels(
        self,
        num_labels=2,
        num_layers=2,
        hidden_channels=64,
    ):

        # set num_labels to 1 for binary classification output size
        num_labels = 1 if num_labels == 2 else num_labels

        # last output channel size should always be num_labels
        if isinstance(hidden_channels, int):
            out_channels = [hidden_channels] * (num_layers - 1) + [num_labels]
        else:
            out_channels = hidden_channels
            if len(out_channels) == num_layers:
                assert out_channels[-1] == num_labels
            elif len(out_channels) == (num_layers - 1):
                out_channels += [num_labels]
            else:
                raise ValueError(
                    "hidden_channels should either be an int or a "
                    "sequence with length in (num_layers, num_layers-1)"
                )

        # first block with encoder output as input size
        in_channels = [self.encoder.config.hidden_size] + out_channels[:-1]

        return in_channels, out_channels
