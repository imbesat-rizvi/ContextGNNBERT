def mask_averaging(seq, mask):
    r"""
    averages vectors in seq according to mask which may have one less dimension
    corresponding to the vector length i.e. seq is BxTxD and mask can either be
    BxTxD or BxT
    """

    if mask.shape != seq.shape:
        # expand mask dimension and repeat mask value
        # i.e. convert mask from BxT to BxTxD
        mask = mask.unsqueeze(-1).expand(-1, -1, seq.shape[-1])

    masked_seq_sum = (mask * seq).sum(dim=1)  # masked sum across T
    masked_avg = masked_seq_sum / mask.sum(dim=1)

    return masked_avg
