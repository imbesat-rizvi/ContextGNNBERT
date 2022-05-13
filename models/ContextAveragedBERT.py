import torch
from transformers.modeling_outputs import SequenceClassifierOutput

from .FCNBERT import FCNBERT
from .utils import mask_averaging


class ContextAveragedBERT(FCNBERT):
    def __init__(
        self,
        encoder,
        num_labels=2,
        trainable_encoder=True,
        num_context_types=1,
        num_layers=2,
        hidden_channels=64,
        dropout=0.2,
        lin_kwargs={},
        non_linearity="ReLU",
    ):

        super(ContextAveragedBERT, self).__init__(
            encoder,
            num_labels=num_labels,
            trainable_encoder=trainable_encoder,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            dropout=dropout,
            lin_kwargs=lin_kwargs,
            non_linearity=non_linearity,
        )

        context_mlp = [
            torch.nn.Linear(
                in_features=encoder.config.hidden_size * (num_context_types + 1),
                out_features=encoder.config.hidden_size,
                **lin_kwargs,
            ),
            torch.nn.__dict__[non_linearity](),
        ]

        if dropout:
            context_mlp.append(torch.nn.Dropout(p=dropout))

        self.context_mlp = torch.nn.Sequential(*context_mlp)

    def forward(
        self, input_ids, attention_mask, token_type_ids, context_mask, labels=None
    ):
        r"""placeholder labels argument is provided for the compatibility with
        the huggingface dataset and Trainer api's compute_metrics."""

        seq_out, pooled_out = self.encoder(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False,
        )

        mask = context_mask * (token_type_ids == 0)
        seq1_averaged_context = mask_averaging(seq_out, mask)
        pooled_cat_context = torch.cat([pooled_out, seq1_averaged_context], dim=-1)

        paired_seq = torch.any(token_type_ids)
        if paired_seq:
            mask = context_mask * (token_type_ids == 1)
            seq2_averaged_context = mask_averaging(seq_out, mask)
            pooled_cat_context = torch.cat(
                [pooled_cat_context, seq2_averaged_context], dim=-1
            )

        output = self.fcn(self.context_mlp(pooled_cat_context))
        return SequenceClassifierOutput(logits=output)
