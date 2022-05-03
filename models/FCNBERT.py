import torch
from transformers.modeling_outputs import SequenceClassifierOutput
from .EncoderBERT import EncoderBERT


class FCNBERT(EncoderBERT):
    r"""Fully-connected Network on top of BERT Encoder"""

    def __init__(
        self, 
        encoder, 
        num_labels=2,
        trainable_encoder=False,
        num_layers=2,
        hidden_channels=64,
        dropout=0.2,
        lin_kwargs={},
        non_linearity="ReLU",
    ):
        
        super(FCNBERT, self).__init__(encoder, num_labels, trainable_encoder)

        self.fcn = self.create_fcn(
            num_labels=num_labels,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
            lin_kwargs=lin_kwargs,
            dropout=dropout,
            non_linearity=non_linearity,
        )

    
    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        r"""placeholder labels argument is provided for the compatibility with
        the huggingface dataset and Trainer api's compute_metrics."""
        
        seq_out, pooled_out = self.encoder(
            input_ids, 
            attention_mask=attention_mask, 
            token_type_ids=token_type_ids,
            return_dict=False,
        )

        output = self.fcn(pooled_out)
        return SequenceClassifierOutput(logits=output)


    def create_fcn(
        self,
        num_labels=2,
        num_layers=2,
        hidden_channels=64,
        lin_kwargs={},
        dropout=0.2,
        non_linearity="ReLU",
    ):

        in_channels, out_channels = self.get_in_out_channels(
            num_labels=num_labels,
            num_layers=num_layers,
            hidden_channels=hidden_channels,
        )

        fcn = []

        for i, in_ch, out_ch in zip(
            range(1, num_layers+1), in_channels, out_channels
        ):

            fcn.append(
                torch.nn.Linear(in_features=in_ch, out_features=out_ch, **lin_kwargs)
            )

            if i < num_layers:
                fcn.append(torch.nn.__dict__[non_linearity]())
                if dropout:
                    fcn.append(torch.nn.Dropout(p=dropout))

        return torch.nn.Sequential(*fcn)