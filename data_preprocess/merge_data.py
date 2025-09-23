from pathlib import Path

import pandas as pd


def load_dataset(path: Path):
    return pd.read_csv(path, index_col=0)


def main():
    data_dir = Path(__file__).resolve().parent.parent / "data"
    train_path = data_dir / "fraudTrain.csv"
    test_path = data_dir / "fraudTest.csv"
    output_path = data_dir / "fraud_detection_data.csv"

    train_df = load_dataset(train_path)
    test_df = load_dataset(test_path)

    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    combined_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
