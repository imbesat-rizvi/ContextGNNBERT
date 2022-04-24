import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
)

from .post_process import to_labels

def compute_aprfbeta(eval_pred, acc_kwargs={}, prfbeta_kwargs={}):
    model_out, labels = eval_pred
    pred = to_labels(model_out)
    if model_out.shape[1] == 1:
        prfbeta_kwargs["average"] = "binary"
    
    acc = accuracy_score(y_true=labels, y_pred=pred, **acc_kwargs)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true=labels, y_pred=pred, **prfbeta_kwargs
    )

    score = dict(zip(
        ("accuracy", "precision", "recall", "f1"), 
        (acc, p, r, f1),
    ))

    if "beta" in prfbeta_kwargs:
        beta = prfbeta_kwargs["beta"]
        score[f"f{beta}"] = score.pop("f1")

    return score