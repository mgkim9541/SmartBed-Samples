#!/usr/bin/env python3
"""Entry point to run the full  pipeline from the command line."""

import argparse
from pathlib import Path
import pandas as pd

from Codes.preprocessing import preprocess
from Codes.models import train_model
from Codes.evaluate import evaluate

def main():
    parser = argparse.ArgumentParser(description="Run ML pipeline.")
    parser.add_argument('--data', type=Path, required=True, help='Path to CSV file.')
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs.')
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    X_train, X_test, y_train, y_test = preprocess(df)

    model = train_model(X_train, y_train, epochs=args.epochs)
    metrics = evaluate(model, X_test, y_test)

    print("\n=== Evaluation ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

if __name__ == '__main__':
    main()
