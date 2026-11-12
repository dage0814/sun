# AST-GTN

PyTorch code for next-day stock direction prediction using financial features, news and stock relations.

The model combines GP factor search, industry-conditioned news encoding, dual-path cross-attention, graph attention, and Transformer/TCN temporal branches.

## Setup

Requires Python 3.10 or later.

```bash
pip install -r requirements.txt
```

The text encoder uses `ProsusAI/finbert`. Pretrained weights and financial datasets are not included.

## Training

```bash
python train.py --data data/bundle.pt --output results --seed 42
```

Prepare an aligned tensor bundle before training. Raw-data acquisition and bundle construction are not included. See `astgtn/data.py` and `astgtn/contract.py` for input fields and timestamp checks. GP factor search and IC selection run before bundle construction.

Training exports the selected checkpoint and validation/test predictions. `bootstrap.py` computes paired MCC comparisons from prediction CSV files.

## Files

- `astgtn/`: factors, news encoding, fusion, model and training.
- `plots/`: plots from supplied experiment summaries and analytical functions.
- `tests/`: unit and integration tests using synthetic inputs.

```bash
python -m unittest discover -s tests
```
