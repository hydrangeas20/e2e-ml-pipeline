import os
import yaml
import pandas as pd

# Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))["preprocess"]

def preprocess(input_path: str, output_path: str) -> None:
    data = pd.read_csv(input_path)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Header is kept downstream, so code can find "Outcome"
    data.to_csv(output_path, index=False)

    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    preprocess(params["input"], params["output"])
