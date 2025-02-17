import argparse
import json
from pathlib import Path
from pprint import pprint
from onnx_helper import SyndromeDetectionModel

parser = argparse.ArgumentParser(
    description="Run Syndrome Detection Model on a directory of images"
)
parser.add_argument(
    "--input_dir", type=str, required=True, help="Path to directory containing images"
)
parser.add_argument(
    "--output_dir", type=str, required=True, help="Path to save results"
)
args = parser.parse_args()

input_dir = Path(args.input_dir)
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

model = SyndromeDetectionModel("syndrome_detection_model.onnx")

outputs = model.predict_dir(input_dir)

pprint(outputs)

output_file = output_dir / "results.json"
with open(output_file, "w") as f:
    json.dump(outputs, f, indent=4)

print(f"Results saved to {output_file}")
