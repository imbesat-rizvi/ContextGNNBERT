import numpy as np
from functools import partial
from pathlib import Path
from sklearn.utils.class_weight import compute_class_weight

from datasets import load_dataset, ClassLabel

from data_utils.hf_utils import (
    hstack_cols, 
    get_tokenizer_encoder,
    tokenize_dataset,
)

import ContextMasker
from models import ContextGNNBERT
from models.utils import HFTrainer, compute_aprfbeta, to_labels

import torch
from torch.optim import AdamW
from transformers import (
    get_linear_schedule_with_warmup,
    EarlyStoppingCallback,
    TrainingArguments,
)

from arguments import parser
import wandb


def main(
    dataset_name="super_glue",
    config_name="boolq",
    cols_for_context=["passage", "question"],
    context_masker="TFIDFContextMasker",
    context_mask_fn_kwargs={"percentile_cutoff": (75, 50)},
    truncation_strategy="only_first",
    encoder_model="bert-base-uncased",
    gnn_class="GATv2Conv",
    gnn_kwargs=dict(heads=1),
    gnn_block_dropout=0.2,
    num_gnn_blocks=3,
    hidden_channels=[256,64],
    num_train_epochs=50,
    eval_steps=100,
    batch_size=32,
    lr=1e-3,
    adam_eps=1e-8,
    num_warmup_steps=0,
    early_stopping_patience=7,
    early_stopping_threshold=1e-3,
    no_class_weight=False,
):

    # rename the HF dataset label col to labels as HF pipelines expect this name
    label_col = "labels"
    num_labels = None
    cols_to_exl_in_model_inp = []
    dataset = load_dataset(dataset_name, config_name)

    for feat_name, feat_class in dataset["train"].features.items():
        if isinstance(feat_class, ClassLabel):
            dataset = dataset.rename_column(feat_name, label_col)
            num_labels = feat_class.num_classes
        else:
            cols_to_exl_in_model_inp.append(feat_name)

    if isinstance(cols_for_context, str):
        context_corpus_col = cols_for_context
    else:
        context_corpus_col = "context_corpus"

    tokenizer, encoder = get_tokenizer_encoder(encoder_model=encoder_model)
    max_tokenized_length = encoder.config.max_position_embeddings

    dataset = tokenize_dataset(
        dataset,
        tokenizer=tokenizer,
        tokenizing_fields=cols_for_context,
        truncation=truncation_strategy,
        padding="max_length",
        max_length=max_tokenized_length,
    )

    context_corpus = dataset["train"]
    if not isinstance(cols_for_context, str):
        context_corpus = hstack_cols(
            context_corpus, cols=cols_for_context, stacked_col_name=context_corpus_col
        )

    context_masker = ContextMasker.__dict__[context_masker](
        context_corpus[context_corpus_col], tokenizer=tokenizer
    )

    dataset = context_masker.insert_context_mask(
        dataset, cols=cols_for_context, **context_mask_fn_kwargs
    )

    dataset = dataset.remove_columns(cols_to_exl_in_model_inp)
    dataset.set_format("torch")

    model = ContextGNNBERT(
        encoder=encoder,
        num_labels=num_labels,
        gnn_class=gnn_class,
        gnn_kwargs=gnn_kwargs,
        gnn_block_dropout=gnn_block_dropout,
        num_gnn_blocks=num_gnn_blocks,
        hidden_channels=hidden_channels,
    )

    run_name = f"{dataset_name}-{config_name}-{encoder_model}"
    trainer_dir = Path("trainer_outputs")
    trainer_dir.mkdir(parents=True, exist_ok=True)
    output_dir = trainer_dir/run_name
    
    train_args = TrainingArguments(
        run_name=run_name,
        report_to="wandb",
        num_train_epochs=num_train_epochs,
        no_cuda=(not torch.cuda.is_available()),
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=eval_steps,  # defaults to logging_steps if not provided
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        metric_for_best_model="eval_loss",
        greater_is_better=False,  # as smaller loss is better
        remove_unused_columns=False,
        logging_strategy="steps",
        logging_steps=eval_steps,
        load_best_model_at_end=True,
        # save_strategy must be same as evaluation_strategy
        # and save_steps must be a round_multiple of eval_steps
        # if load_best_model_at_end is True
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=10,  # deletes older checkpoints on reaching this limit
    )

    num_train_steps = int(np.ceil(len(dataset["train"]) * num_train_epochs / batch_size))
    optimizer = AdamW(model.parameters(), lr=lr, eps=adam_eps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=early_stopping_threshold,
    )

    compute_metrics = partial(
        compute_aprfbeta, 
        prfbeta_kwargs=dict(average="macro"),
    )

    trainer = HFTrainer(
        model=model,
        args=train_args,
        optimizers=(optimizer, scheduler),
        train_dataset=dataset["train"].select(range(10)),
        eval_dataset=dataset["validation"].select(range(10)),
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],
    )

    binary_class = num_labels==2
    labels = np.array(dataset["train"][label_col])
    
    if binary_class:
        # pos_weight of torch's BCEWithLogitsLoss expects 
        # it in this way i.e. count(neg_class) / count(pos_class)
        # see the example explanation at:
        # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
        class_counts = np.bincount(labels)
        class_weight = class_counts[0]/class_counts[1]
    else:
        class_weight = compute_class_weight(
            "balanced", classes=np.unique(labels), y=labels
        )
    
    if no_class_weight:
        trainer.register_loss_fn(binary_class, weight=None)
    else:
        trainer.register_loss_fn(binary_class, weight=class_weight)

    trainer.train()

    dataset_for_metrics = ["train", "validation"]
    if not (-1 in dataset["test"][label_col]):
        dataset_for_metrics.append("test")
    else:
        # nothing to compare, write predictions to a file
        activations = trainer.predict(
            dataset["test"].select(range(10)).remove_columns(label_col)
        ).predictions
        
        pred_labels = to_labels(activations)
        np.savetxt(
            output_dir/"predictions.csv", 
            pred_labels, 
            fmt="%d",
            header="predictions",
            comments="",
        )

    pred_metrics = {
        i: trainer.predict(dataset[i].select(range(10)), metric_key_prefix=i).metrics
        for i in dataset_for_metrics
    }

    wandb.log(pred_metrics)
    print(pred_metrics)


if __name__ == "__main__":
    args = parser.parse_args()

    if len(args.hidden_channels) == 1:
        args.hidden_channels = args.hidden_channels[0]
    if len(args.cols_for_context) == 1:
        args.cols_for_context = args.cols_for_context[0]

    wandb.login() # prompts for logging in if not done already
    wandb.init(
        project="ContextGNNBERT",
        name=f"{args.dataset_name}-{args.config_name}-{args.encoder_model}",
        tags=["ContextGNNBERT", args.dataset_name, args.config_name, args.encoder_model],
        group="ContextGNNBERT",
        entity="pensieves",
        config=args,
    )

    main(
        dataset_name=args.dataset_name,
        config_name=args.config_name,
        cols_for_context=args.cols_for_context,
        context_masker=args.context_masker,
        context_mask_fn_kwargs=args.context_mask_fn_kwargs,
        truncation_strategy=args.truncation_strategy,
        encoder_model=args.encoder_model,
        gnn_class=args.gnn_class,
        gnn_kwargs=args.gnn_kwargs,
        gnn_block_dropout=args.gnn_block_dropout,
        num_gnn_blocks=args.num_gnn_blocks,
        hidden_channels=args.hidden_channels,
        num_train_epochs=args.num_train_epochs,
        eval_steps=args.eval_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        adam_eps=args.adam_eps,
        num_warmup_steps=args.num_warmup_steps,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=args.early_stopping_threshold,
        no_class_weight=args.no_class_weight,
    )