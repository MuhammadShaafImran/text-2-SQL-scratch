# Text-to-SQL analysis

## Starter statistics

| Split | Pairs | Source mean | Source max | Target mean | Target max |
|---|---:|---:|---:|---:|---:|
| train | 56,355 | 41.55 | 221 | 12.77 | 63 |
| dev | 8,421 | 41.49 | 166 | 12.79 | 42 |
| test | 15,878 | 41.67 | 259 | 12.93 | 44 |

## Model facts

- Trainable parameters: **7,585,600**.
- Configuration: d_model=256, heads=4, encoder/decoder layers=3/3, feed-forward size=1024, dropout=0.1, post-norm LayerNorm.
- `starter/check_starter.py` reports `(B, S)` and `(B, T)` token batches, followed by `(B, S, 256)` and `(B, T-1, 256)` embeddings.

## Positional encoding

The heatmap shows alternating sinusoidal patterns across positions and embedding dimensions. Low-frequency dimensions change slowly while high-frequency dimensions change rapidly, giving every position a distinct representation that can be extrapolated beyond training positions.

## Cross-attention

Generated figure: `cross_attention_dev.png`. It averages the heads from the final decoder layer for one dev example.
