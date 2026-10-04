import torch

from model.transformer import Transformer
from starter.embeddings import TokenEmbedding
from starter.tokenizer import load_tokenizer, PAD_ID
from decode import greedy_decode, parse_prediction, query_to_sql

CHECKPOINT_PATH = "best_model.pt"
TOKENIZER_PATH = "data/sql_sp.model"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sp = load_tokenizer(TOKENIZER_PATH)
checkpoint = torch.load( CHECKPOINT_PATH, map_location=device, weights_only=False)
config = checkpoint["config"]
vocab_size = sp.get_piece_size()

model = Transformer(
    num_layers=config["num_layers"],
    d_model=config["d_model"],
    heads=config["heads"],
    token_embedding=TokenEmbedding(
        vocab_size=vocab_size,
        d_model=config["d_model"],
        pad_id=PAD_ID,
    ),
    ffn_dim=config["d_ff"],
    max_len=512,
    dropout=config["dropout"],
    vocab_size=vocab_size,
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

question = "what position does the player who played for butler cc (ks) play?"
header = [
    "player",
    "no.",
    "nationality",
    "position",
    "years in toronto",
    "school/club team",
]

columns = " ".join(f"<c{i}> {name}" for i, name in enumerate(header))
source = f"{question.strip()} <sep> {columns}".lower()
src_ids = torch.tensor([sp.encode(source, out_type=int)],dtype=torch.long,device=device)
token_ids, prediction = greedy_decode(model,src_ids,sp,device)
parsed = parse_prediction(prediction)
sql = query_to_sql(parsed, header) if parsed is not None else None

print("Generated target:", prediction)
print("Parsed prediction:", parsed)
print("SQL:", sql)