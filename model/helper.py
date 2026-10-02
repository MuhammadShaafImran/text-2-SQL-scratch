import torch

def create_padding_mask(ids, pad_id):
    """
    ids: [B, L]

    Returns:
        [B, 1, 1, L]
    """
    return (ids != pad_id).unsqueeze(1).unsqueeze(2)


def create_causal_mask(length, device):
    """
    Returns:
        [1, 1, L, L]

    Allows each position to attend only to
    itself and previous positions.
    """
    mask = torch.tril(
        torch.ones(
            length,
            length,
            dtype=torch.bool,
            device=device
        )
    )

    return mask.unsqueeze(0).unsqueeze(0)


def create_target_mask(tgt_ids, pad_id):
    """
    Combines:
        - target padding mask
        - causal mask

    Returns:
        [B, 1, T, T]
    """

    batch_size, target_len = tgt_ids.shape

    # [B, 1, 1, T]
    padding_mask = create_padding_mask(
        tgt_ids,
        pad_id
    )

    # [1, 1, T, T]
    causal_mask = create_causal_mask(
        target_len,
        tgt_ids.device
    )

    # Broadcasting:
    # [B, 1, 1, T]
    # &
    # [1, 1, T, T]
    # =
    # [B, 1, T, T]

    return padding_mask & causal_mask
