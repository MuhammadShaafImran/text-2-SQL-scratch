import json
import re
import torch

from starter.tokenizer import (load_tokenizer, BOS_ID, EOS_ID, PAD_ID)
from starter.data_prep import (AGG_OPS,COND_OPS)

MAX_DECODE_LEN = 64
BEAM_SIZE = 4

def is_eos(token_id):
    return token_id == EOS_ID

def decode_ids(sp, ids):
    return sp.decode(ids)


@torch.no_grad()
def greedy_decode(
    model,
    src_ids,
    sp,
    device,
    max_len=MAX_DECODE_LEN,
):
    """
    Autoregressively generate target tokens.

    Stops when:
        - EOS is generated
        - max_len is reached
    """

    model.eval()

    src_ids = src_ids.to(device)

    # Start with <BOS>
    generated = torch.tensor(
        [[BOS_ID]],
        dtype=torch.long,
        device=device,
    )

    for _ in range(max_len):

        logits, _ = model(
            src_ids,
            generated,
        )

        # Last decoder position
        next_token_logits = logits[:, -1, :]

        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True,
        )

        generated = torch.cat(
            [generated, next_token],
            dim=1,
        )

        if next_token.item() == EOS_ID:
            break

    token_ids = generated[0].tolist()

    # Remove BOS
    if token_ids and token_ids[0] == BOS_ID:
        token_ids = token_ids[1:]

    # Remove EOS
    if EOS_ID in token_ids:
        token_ids = token_ids[:token_ids.index(EOS_ID)]

    return token_ids, sp.decode(token_ids)


def parse_prediction(text):
    """
    Convert:

        select count <c3> where <c1> = kim manners

    into:

        {
            "sel": 3,
            "agg": 3,
            "conds": [
                [1, 0, "kim manners"]
            ]
        }

    Returns None if parsing fails.
    """

    text = text.strip().lower()

    if not text.startswith("select "):
        return None

    # --------------------------------------------------------
    # Split SELECT and WHERE
    # --------------------------------------------------------

    parts = re.split(
        r"\s+where\s+",
        text,
        maxsplit=1,
    )

    select_part = parts[0]

    if len(parts) == 2:
        where_part = parts[1]
    else:
        where_part = ""

    # --------------------------------------------------------
    # Parse SELECT
    # --------------------------------------------------------

    select_tokens = select_part.split()

    if len(select_tokens) < 2:
        return None

    # Possible forms:
    #
    # select <c3>
    # select count <c3>
    #

    if select_tokens[1] in [
        "max",
        "min",
        "count",
        "sum",
        "avg",
    ]:

        agg_name = select_tokens[1]

        agg_map = {
            "max": 1,
            "min": 2,
            "count": 3,
            "sum": 4,
            "avg": 5,
        }

        agg = agg_map[agg_name]

        if len(select_tokens) != 3:
            return None

        column_token = select_tokens[2]

    else:

        agg = 0

        if len(select_tokens) != 2:
            return None

        column_token = select_tokens[1]

    # --------------------------------------------------------
    # Parse SELECT column
    # --------------------------------------------------------

    match = re.fullmatch(
        r"<c(\d+)>",
        column_token,
    )

    if match is None:
        return None

    sel = int(match.group(1))

    # --------------------------------------------------------
    # Parse WHERE conditions
    # --------------------------------------------------------

    conditions = []

    if where_part:

        # "and" separates conditions.
        condition_parts = re.split(
            r"\s+and\s+",
            where_part,
        )

        for condition in condition_parts:

            tokens = condition.strip().split()

            if len(tokens) < 3:
                return None

            column_token = tokens[0]
            operator = tokens[1]

            value_tokens = tokens[2:]

            # Column must be <cK>
            match = re.fullmatch(
                r"<c(\d+)>",
                column_token,
            )

            if match is None:
                return None

            col = int(match.group(1))

            # Operator must be valid
            if operator not in COND_OPS:
                return None

            op = COND_OPS.index(operator)

            # Value must exist
            if not value_tokens:
                return None

            value = " ".join(value_tokens)

            conditions.append(
                [col, op, value]
            )

    return {
        "sel": sel,
        "agg": agg,
        "conds": conditions,
    }


def query_to_sql(query, header):

    if query is None:
        return None

    sel = query["sel"]
    agg = query["agg"]
    conditions = query["conds"]

    if sel < 0 or sel >= len(header):
        return None

    # --------------------------------------------------------
    # SELECT
    # --------------------------------------------------------

    column_name = header[sel]

    if agg == 0:
        sql = f"SELECT {column_name}"
    else:

        if agg < 0 or agg >= len(AGG_OPS):
            return None

        aggregation = AGG_OPS[agg].strip()

        sql = (
            f"SELECT {aggregation}"
            f"({column_name})"
        )

    # --------------------------------------------------------
    # WHERE
    # --------------------------------------------------------

    if conditions:

        where_parts = []

        for col, op, value in conditions:

            if col < 0 or col >= len(header):
                return None

            if op < 0 or op >= len(COND_OPS):
                return None

            column_name = header[col]
            operator = COND_OPS[op]

            # Quote string values.
            value_sql = f"'{value}'"

            where_parts.append(
                f"{column_name} {operator} {value_sql}"
            )

        sql += " WHERE " + " AND ".join(
            where_parts
        )

    return sql

@torch.no_grad()
def beam_search(
model,
src_ids,
device,
beam_size=BEAM_SIZE,
max_len=MAX_DECODE_LEN,
):
    """
    Beam search with beam size 4.

    Returns the best token-ID sequence.
    """

    model.eval()

    src_ids = src_ids.to(device)

    beams = [
        (
            [BOS_ID],
            0.0,
        )
    ]

    completed = []

    for _ in range(max_len):

        candidates = []

        for tokens, score in beams:

            if tokens[-1] == EOS_ID:
                completed.append(
                    (tokens, score)
                )
                continue

            decoder_input = torch.tensor(
                [tokens],
                dtype=torch.long,
                device=device,
            )

            logits, _ = model(
                src_ids,
                decoder_input,
            )

            next_logits = logits[:, -1, :]

            log_probs = torch.log_softmax(
                next_logits,
                dim=-1,
            )

            values, indices = torch.topk(
                log_probs,
                beam_size,
                dim=-1,
            )

            for i in range(beam_size):

                token = indices[0, i].item()
                token_score = values[0, i].item()

                candidates.append(
                    (
                        tokens + [token],
                        score + token_score,
                    )
                )

        if not candidates:
            break

        candidates.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        beams = candidates[:beam_size]

        # If every beam has ended, stop.
        if all(
            tokens[-1] == EOS_ID
            for tokens, _ in beams
        ):
            completed.extend(beams)
            break

    if completed:
        completed.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        best_tokens = completed[0][0]

    else:
        best_tokens = beams[0][0]

    # Remove BOS
    if best_tokens and best_tokens[0] == BOS_ID:
        best_tokens = best_tokens[1:]

    # Remove EOS
    if EOS_ID in best_tokens:
        best_tokens = best_tokens[
            :best_tokens.index(EOS_ID)
        ]

    return best_tokens