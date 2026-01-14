"""Tokenization for modular addition.


For "BOS_A_B_SEP" format:
    - Token IDs: numbers 0..p-1 map to 0..p-1
    - BOS = p
    - SEP = p+1
    - vocab_size = p + 2
    - sequence: [BOS, a, b, SEP]
    - predict c at position 3 (SEP position)

For "A_B_EQ" format:
    - Token IDs: numbers 0..p-1 map to 0..p-1
    - EQ = p
    - vocab_size = p + 1
    - sequence: [a, b, EQ]
    - predict c at position 2 (EQ position)
"""

from typing import Any

import torch


def get_vocab(p: int, format: str) -> dict[str, Any]:
    """Get vocabulary information for the given format.

    Args:
        p: Prime modulus.
        format: Token format ("BOS_A_B_SEP" or "A_B_EQ").

    Returns:
        Dictionary containing:
            - vocab_size: Total vocabulary size
            - special_tokens: Dict mapping token names to IDs
            - seq_len: Sequence length for this format
            - pred_pos: Position index for prediction

    Raises:
        ValueError: If format is invalid.
    """
    if format == "BOS_A_B_SEP":
        return {
            "vocab_size": p + 2,
            "special_tokens": {
                "BOS": p,
                "SEP": p + 1,
            },
            "seq_len": 4,
            "pred_pos": 3,  # Predict at SEP position
        }
    elif format == "A_B_EQ":
        return {
            "vocab_size": p + 1,
            "special_tokens": {
                "EQ": p,
            },
            "seq_len": 3,
            "pred_pos": 2,  # Predict at EQ position
        }
    else:
        raise ValueError(f"Invalid format: {format}. Must be 'BOS_A_B_SEP' or 'A_B_EQ'.")


def encode_example(
    a: int,
    b: int,
    p: int,
    format: str,
) -> tuple[torch.Tensor, int]:
    """Encode a single example as tokens.

    Args:
        a: First operand (0 <= a < p).
        b: Second operand (0 <= b < p).
        p: Prime modulus.
        format: Token format ("BOS_A_B_SEP" or "A_B_EQ").

    Returns:
        Tuple of (tokens, target) where:
            - tokens: LongTensor of shape [seq_len]
            - target: The correct answer ((a + b) mod p)

    Raises:
        ValueError: If format is invalid or operands out of range.
    """
    if not (0 <= a < p and 0 <= b < p):
        raise ValueError(f"Operands must be in [0, {p-1}], got a={a}, b={b}")

    vocab = get_vocab(p, format)
    target = (a + b) % p

    if format == "BOS_A_B_SEP":
        tokens = torch.tensor(
            [vocab["special_tokens"]["BOS"], a, b, vocab["special_tokens"]["SEP"]],
            dtype=torch.long
        )
    elif format == "A_B_EQ":
        tokens = torch.tensor(
            [a, b, vocab["special_tokens"]["EQ"]],
            dtype=torch.long
        )
    else:
        raise ValueError(f"Invalid format: {format}")

    return tokens, target


def decode_tokens(
    tokens: torch.Tensor,
    p: int,
    format: str,
) -> str:
    """Decode tokens back to a readable string (for debugging).

    Args:
        tokens: LongTensor of token IDs.
        p: Prime modulus.
        format: Token format ("BOS_A_B_SEP" or "A_B_EQ").

    Returns:
        String representation of the tokens.
    """
    vocab = get_vocab(p, format)
    special_tokens = vocab["special_tokens"]

    # Create reverse mapping
    id_to_name = {v: k for k, v in special_tokens.items()}

    parts = []
    for token_id in tokens.tolist():
        if token_id in id_to_name:
            parts.append(f"<{id_to_name[token_id]}>")
        elif 0 <= token_id < p:
            parts.append(str(token_id))
        else:
            parts.append(f"<UNK:{token_id}>")

    return " ".join(parts)


def get_number_token_range(p: int) -> tuple[int, int]:
    """Get the range of token IDs that represent numbers.

    Args:
        p: Prime modulus.

    Returns:
        Tuple of (start, end) where number tokens are in [start, end).
    """
    return (0, p)
