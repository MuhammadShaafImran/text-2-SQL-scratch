import re
import torch

from starter.tokenizer import (BOS_ID, EOS_ID)
from starter.data_prep import (AGG_OPS,COND_OPS)

MAX_DECODE_LEN = 64
BEAM_SIZE = 4

# GNMT-style length penalty exponent. Without it the raw sum of log-probs
# always favours the shortest hypothesis, so an immediate </s> (an empty
# target string) outranks every complete query and the parser rejects it.
BEAM_LENGTH_PENALTY = 0.7

# A completed hypothesis needs at least this many content tokens to be a
# parseable query ("select <c0>" is already 2).
MIN_CONTENT_TOKENS = 2


def is_eos(token_id):
    return token_id == EOS_ID


def beam_length_penalty(num_generated_tokens, alpha=BEAM_LENGTH_PENALTY):
    """Normalise a cumulative log-prob by sequence length (GNMT, Eq. 14)."""
    if num_generated_tokens <= 0:
        return float("-inf")
    return num_generated_tokens ** alpha

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

        # "and" separates conditions -- but only when a column token and an
        # operator follow it.  Gold values may themselves contain "and"
        # (e.g. "= karen felix and don woodard"), and WikiSQL stores that as
        # ONE condition, so a blind split would make it unparseable.
        condition_parts = re.split(
            r"\s+and\s+(?=<c\d+>\s*(?:=|>|<))",
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

    Hypotheses are ranked by length-normalised log-probability
    (score / len**0.7). Raw cumulative log-prob is also carried so the
    ranking can be reproduced.

    Returns the best token-ID sequence (BOS/EOS stripped).
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

    def normalised(tokens, score):
        # Exclude BOS from the length so a single generated token counts as 1.
        generated = max(len(tokens) - 1, 1)
        penalty = beam_length_penalty(generated)
        return score / penalty

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
            key=lambda x: normalised(x[0], x[1]),
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
            key=lambda x: normalised(x[0], x[1]),
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

    # An empty or near-empty hypothesis (e.g. just "</s>") is never a valid
    # query. Fall back to the best-scoring completed hypothesis long enough to
    # parse, so decode output is never "".
    if len(best_tokens) < MIN_CONTENT_TOKENS and completed:

        fallback = None
        for tokens, score in completed:
            stripped = tokens[1:] if tokens[:1] == [BOS_ID] else list(tokens)
            if EOS_ID in stripped:
                stripped = stripped[: stripped.index(EOS_ID)]
            if len(stripped) < MIN_CONTENT_TOKENS:
                continue
            # Rank by the same normalised score used above, keeping each
            # candidate's own score rather than the current best's.
            if fallback is None or normalised(tokens, score) > normalised(
                fallback[0], fallback[1]
            ):
                fallback = (tokens, score)

        if fallback is not None:
            best_tokens = fallback[0]
            if best_tokens and best_tokens[0] == BOS_ID:
                best_tokens = best_tokens[1:]
            if EOS_ID in best_tokens:
                best_tokens = best_tokens[: best_tokens.index(EOS_ID)]

    return best_tokens