import torch
import torch.nn as nn
from pathlib import Path

from model.transformer import Transformer
from starter.dataset import make_loader
from starter.embeddings import TokenEmbedding
from starter.tokenizer import load_tokenizer, PAD_ID

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VOCAB_SIZE = None
D_MODEL = 256
HEADS = 4
D_FF = 1024
NUM_LAYERS = 3
MAX_LEN = 512
DROPOUT = 0.1

BATCH_SIZE = 48
ADDITIONAL_EPOCHS = 5

WARMUP_STEPS = 4000
LABEL_SMOOTHING = 0.1
MAX_GRAD_NORM = 1.0

LEARNING_RATE_EPS = 1e-9
BETA1 = 0.9
BETA2 = 0.98

BEST_CHECKPOINT = "best_model.pt"
LAST_CHECKPOINT = "last_model.pt"
RESUME_CHECKPOINT = LAST_CHECKPOINT

TRAIN_PATH = "data/train_pairs.jsonl"
DEV_PATH = "data/dev_pairs.jsonl"
SP_MODEL_PATH = "data/sql_sp.model"

def transformer_lr(step, d_model, warmup_steps):
    """
    Attention Is All You Need, Equation 3:

        lrate =
            d_model^(-0.5)
            * min(
                step^(-0.5),
                step * warmup_steps^(-1.5)
            )
    """

    if step == 0:
        return 0.0

    return (
        d_model ** -0.5
        * min(
            step ** -0.5,
            step * warmup_steps ** -1.5
        )
    )

@torch.no_grad()
def evaluate( model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    batches = 0
    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)
        decoder_input = tgt[:, :-1]
        target = tgt[:, 1:]
        logits, _ = model(src,decoder_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        total_loss += loss.item()
        batches += 1

    return total_loss / batches

def train_one_epoch(model, loader, criterion, optimizer, device, global_step):
    model.train()
    total_loss = 0.0
    batches = 0
    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)
        decoder_input = tgt[:, :-1]
        target = tgt[:, 1:]

        global_step += 1
        lr = transformer_lr( global_step, D_MODEL, WARMUP_STEPS)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        logits, _ = model(src, decoder_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), target.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        total_loss += loss.item()
        batches += 1

    average_loss = total_loss / batches
    current_lr = optimizer.param_groups[0]["lr"]
    return average_loss, global_step, current_lr, grad_norm.item()


def build_checkpoint(
    epoch,
    global_step,
    model,
    optimizer,
    train_loss,
    dev_loss,
    learning_rate,
):
    return {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "dev_loss": dev_loss,
        "learning_rate": learning_rate,
        "config": {
            "d_model": D_MODEL,
            "heads": HEADS,
            "d_ff": D_FF,
            "num_layers": NUM_LAYERS,
            "dropout": DROPOUT,
            "warmup_steps": WARMUP_STEPS,
            "label_smoothing": LABEL_SMOOTHING,
        },
    }


def main():
    print("Device:", DEVICE)
    sp = load_tokenizer(SP_MODEL_PATH)
    global VOCAB_SIZE
    VOCAB_SIZE = sp.get_piece_size()
    print("Vocabulary size:", VOCAB_SIZE)
    
    train_loader = make_loader(TRAIN_PATH, sp, train=True, batch_size=BATCH_SIZE)
    dev_loader = make_loader(DEV_PATH, sp, train=False, batch_size=BATCH_SIZE)
    token_embedding = TokenEmbedding(vocab_size=VOCAB_SIZE,d_model=D_MODEL,pad_id=PAD_ID)
    model = Transformer(
        num_layers=NUM_LAYERS,
        d_model=D_MODEL,
        heads=HEADS,
        token_embedding=token_embedding,
        ffn_dim=D_FF,
        max_len=MAX_LEN,
        dropout=DROPOUT,
        vocab_size=VOCAB_SIZE
    )
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        ignore_index=PAD_ID,
        label_smoothing=LABEL_SMOOTHING
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0,
        betas=(BETA1, BETA2),
        eps=LEARNING_RATE_EPS
    )
    
    checkpoint_path = Path(RESUME_CHECKPOINT)
    if not checkpoint_path.exists() and Path(BEST_CHECKPOINT).exists():
        checkpoint_path = Path(BEST_CHECKPOINT)

    if checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["global_step"]
        best_dev_loss = checkpoint.get(
            "best_dev_loss",
            checkpoint["dev_loss"],
        )
        print(
            f"Resumed from {checkpoint_path}: "
            f"epoch {checkpoint['epoch']}, "
            f"global step {global_step}, "
            f"dev loss {best_dev_loss:.4f}"
        )
    else:
        start_epoch = 1
        global_step = 0
        best_dev_loss = float("inf")
        print(f"No checkpoint found at {RESUME_CHECKPOINT}; starting from scratch.")

    end_epoch = start_epoch + ADDITIONAL_EPOCHS - 1

    for epoch in range(start_epoch, end_epoch + 1):
        train_loss, global_step, current_lr, grad_norm = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            DEVICE,
            global_step,
        )
        dev_loss = evaluate(model, dev_loader, criterion, DEVICE)
        print(
            f"Epoch {epoch:02d}/{end_epoch} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Dev Loss: {dev_loss:.4f} | "
            f"LR: {current_lr:.8f} | "
            f"Grad Norm: {grad_norm:.4f}"
        )

        checkpoint = build_checkpoint(
            epoch,
            global_step,
            model,
            optimizer,
            train_loss,
            dev_loss,
            current_lr,
        )
        checkpoint["best_dev_loss"] = min(best_dev_loss, dev_loss)
        torch.save(checkpoint, LAST_CHECKPOINT)

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(checkpoint,BEST_CHECKPOINT)
            print(f"  Saved checkpoint: {BEST_CHECKPOINT}")
        print(f"  Saved checkpoint: {LAST_CHECKPOINT}")

    print("Training complete.")
    print(
        "Best dev loss:",
        best_dev_loss
    )


if __name__ == "__main__":
    main()