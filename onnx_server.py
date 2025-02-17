import argparse
import csv
import warnings
from typing import TypedDict
from pathlib import Path
from flask_ml.flask_ml_server import MLServer, load_file_as_string
from flask_ml.flask_ml_server.models import (
    DirectoryInput,
    FileResponse,
    InputSchema,
    InputType,
    ResponseBody,
    TaskSchema,
)
from onnx_helper import SyndromeDetectionModel
import torch
import onnxruntime as ort


warnings.filterwarnings("ignore")


def create_transform_case_task_schema() -> TaskSchema:
    input_schema = InputSchema(
        key="input_dataset",
        label="Path to the directory containing all the images",
        input_type=InputType.DIRECTORY,
    )
    output_schema = InputSchema(
        key="output_file",
        label="Path to save results",
        input_type=InputType.DIRECTORY,
    )
    return TaskSchema(inputs=[input_schema, output_schema], parameters=[])


class Inputs(TypedDict):
    input_dataset: DirectoryInput
    output_file: DirectoryInput


class Parameters(TypedDict):
    pass


server = MLServer(__name__)

server.add_app_metadata(
    name="Down Syndrome Detector",
    author="Sachin Thomas",
    version="0.1.0",
    info=load_file_as_string("README.md"),
)

model = SyndromeDetectionModel("syndrome_detection_model.onnx")


@server.route("/predict", task_schema_func=create_transform_case_task_schema)
def give_prediction(inputs: Inputs, parameters: Parameters) -> ResponseBody:
    input_path = inputs["input_dataset"].path
    out = Path(inputs["output_file"].path)
    out = str(out / (f"predictions_" + str(int(torch.rand(1) * 1000)) + ".csv"))

    print(parameters)

    res_list = model.predict_dir(input_path)

    with open(out, mode="w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["image_path", "prediction", "confidence"]
        )
        writer.writeheader()
        writer.writerows(res_list)

    return ResponseBody(FileResponse(path=out, file_type="csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a server.")
    parser.add_argument(
        "--port", type=int, help="Port number to run the server", default=5000
    )
    args = parser.parse_args()

    cuda_available = "CUDAExecutionProvider" in ort.get_available_providers()
    print("CUDA is available." if cuda_available else "CUDA is not available.")

    server.run(port=args.port)
