import torch
from torch.utils.data import Dataset, DataLoader
from starter.tokenizer import read_pairs, PAD_ID, BOS_ID, EOS_ID


class SQLDataset(Dataset):
    """Each item: (source ids, target ids). Target = <s> ... </s>.
    train=True drops over-long pairs.
    train=False keeps EVERY row in file order (needed for evaluation)."""

    def __init__(self, path, sp, train=True, max_src=160, max_tgt=64):
        self.items, skipped = [], 0
        for p in read_pairs(path):
            src = sp.encode(p["src"]) + [EOS_ID]
            tgt = [BOS_ID] + sp.encode(p["tgt"]) + [EOS_ID]
            if train and (len(src) > max_src or len(tgt) > max_tgt):
                skipped += 1
                continue
            self.items.append((src, tgt))
        print(f"{path}: kept {len(self.items)}, skipped {skipped}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    srcs, tgts = zip(*batch)
    S, T = max(map(len, srcs)), max(map(len, tgts))
    src = torch.full((len(batch), S), PAD_ID, dtype=torch.long)
    tgt = torch.full((len(batch), T), PAD_ID, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src[i, : len(s)] = torch.tensor(s)
        tgt[i, : len(t)] = torch.tensor(t)
    return src, tgt  # (B, S), (B, T)


def make_loader(path, sp, train, batch_size=64):
    return DataLoader(
        SQLDataset(path, sp, train=train),
        batch_size=batch_size,
        shuffle=train,
        collate_fn=collate,
    )
