import json
from argparse import ArgumentParser

parser = ArgumentParser(description="Arguments for ContextGNNBERT")
parser.add_argument(
    "--dataset_name",
    default="super_glue:boolq",
    help="huggingface compatible dataset name to be loaded with its config "\
        "name delimited using ':' in case config name is required",
)

parser.add_argument(
    "--label_col",
    help="Label column name to be converted from string type to ClassLabel"\
        "in case of a classification setting",
)

parser.add_argument(
    "--pos_label",
    help="Label name/number denoting positive class for reporting purposes in case"\
        " of binary classification",
)

parser.add_argument(
    "--cols_for_context",
    nargs="+",
    default=["passage", "question"],
    help="columns names in the dataset for context generation. Two column"\
        " names in case of sequence pair classification, else one",
)

parser.add_argument(
    "--context_masker",
    default="TFIDFContextMasker",
    help="Context masker to be used for context generation",
)

parser.add_argument(
    "--context_mask_fn_kwargs",
    type=json.loads,
    default={"percentile_cutoff": (75,50)},
    help="Keyword args to be passed on to the context mask function",
)

parser.add_argument(
    "--truncation_strategy",
    default="only_first",
    # longest_first is equivalent to True, 
    # do_not_truncate is equivalent to False
    choices=["only_first", "only_second", "longest_first", "do_not_truncate"],
    help="truncation strategy for the huggingface tokenizer",
)

parser.add_argument(
    "--encoder_model",
    default="bert-base-uncased",
    help="transformer model to be used as encoder",
)

parser.add_argument(
    "--gnn_class",
    choices=["GATv2Conv", "GATConv", "GCNConv"],
    default="GATv2Conv",
    help="GNN class to be used to initialize the ContextGNNBERT",
)

parser.add_argument(
    "--gnn_kwargs",
    type=json.loads,
    default={"heads": 1},
    help="keyword arguments for the GNN class of the ContextGNNBERT",
)

parser.add_argument(
    "--gnn_block_dropout",
    type=float,
    default=0.2,
    help="dropout for the output from the GNN Block",
)

parser.add_argument(
    "--num_gnn_blocks",
    type=int,
    default=3,
    help="Number of GNN blocks (depth) to be used on top of encoder",
)

parser.add_argument(
    "--hidden_channels",
    nargs="+",
    type=int,
    default=[256, 64],
    help="Number of hidden channels per GNN block. Last layer should "\
        "have hidden channels equal to the number of labels in "\
        " the classification or the list of hidden channels should be "\
        "one less than the num_gnn_blocks in which case last layer's "\
        "hidden channel is automatically selected. Pass a single integer"\
        " if all GNN blocks, except the last one, should have same number"\
        " of hidden channels.",
)

parser.add_argument(
    "--num_train_epochs",
    type=int,
    default=50,
    help="Number of training epochs for the model",
)

parser.add_argument(
    "--eval_steps",
    type=int,
    default=100,
    help="No. of steps at which evaluation is carried out",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=512,
    help="train, eval and test batch size",
)

parser.add_argument(
    "--lr",
    type=float,
    default=1e-3,
    help="learning rate of the AdamW optimizer",
)

parser.add_argument(
    "--adam_eps",
    type=float,
    default=1e-8,
    help="Adam epsilon value for the AdamW optimizer",
)

parser.add_argument(
    "--num_warmup_steps",
    type=int,
    default=0,
    help="Number of warmup steps for the optimizer's scheduler",
)

parser.add_argument(
    "--early_stopping_patience",
    type=int,
    default=7,
    help="Number of steps to watch for before early stopping",
)

parser.add_argument(
    "--early_stopping_threshold",
    type=float,
    default=1e-3,
    help="The threshold to match the early stopping metric/loss"
)

parser.add_argument(
    "--no_class_weight",
    action="store_true",
    help="If specified, the loss function is not evaluated with "\
        "balancing class weights",
)