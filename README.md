# AST-GTN

Next-day stock direction prediction from financial features, news and stock relations. Requires Python 3.10+.

```bash
pip install -r requirements.txt
python train.py --data data/bundle.pt --output results --seed 42
python -m unittest discover -s tests
```

Datasets, pretrained weights and the raw-data-to-bundle pipeline are not included. Input requirements are defined in `astgtn/contract.py`.
