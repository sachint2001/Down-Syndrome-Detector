# Down Syndrome Detector

This project provides a machine learning service for detecting Down syndrome from facial images, utilizing a Flask-ML server and an ONNX model.

This project uses a pre trained model from the following repository: https://github.com/Mitali-Juvekar/Project1_596E. The model has been modified to work with ONNX runtime.

## Setup Instructions ##

1. Clone the repository:
```bash
git clone https://github.com/sachint2001/Down-Syndrome-Detector.git
cd Down-Syndrome-Detector
```

2. Create and activate a virtual environment:
```bash
python -m venv myenv
source myenv/Scripts/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Project Structure ##

* onnx_server.py: Runs a Flask-based ML server that loads the ONNX model and provides an API for predicting Down syndrome from images in a given directory.

* onnx_helper.py: Handles image preprocessing, model inference using ONNX, and post-processing of predictions for Down syndrome detection.

* down_syndrome_cli.py: Provides a command-line interface to run predictions on a directory of images using the ONNX model and saves the results.

* syndrome_detection_model.onnx: A Keras based down syndrome detection model converted to ONNX format, used to predict Down syndrome from input images.

* data/test: Directory containing test images.

## Running the model ##

You can run the model in two ways, as a server to connect to RescueBox or using a CLI

### Method 1: Running as a server for RescueBox

Run the following command to start the Flask-ML server:

```bash
python onnx_server.py
```

You will get the IP address and Port of the server which you can now register with RescueBox to try the model on.

### Method 2: Running it via CLI

The command line interface can be used to test the model. Run the following command:

```bash
python down_syndrome_cli.py --input_dir /path/to/images --output_dir /path/to/save/results
```

Replace "/path/to/images" and "/path/to/save/results" with the directory containing the images you want to test and to store results respectively.

## Process of Exporting to ONNX ##

The trained Keras model was converted to the ONNX (Open Neural Network Exchange) format using tf2onnx. The conversion process involved defining an input signature with a tf.TensorSpec, specifying the expected input shape and data type. The model was then converted using tf2onnx.convert.from_keras(), with an appropriate opset version (16) to ensure compatibility with ONNX runtime. Finally, the serialized ONNX model was written to a file in a single operation.

```bash
import tf2onnx

onnx_model_path = "syndrome_detection_model.onnx"
spec = (tf.TensorSpec((None, 250, 250, 3), tf.float32, name="input"),)
onnx_model, _ = tf2onnx.convert.from_keras(self.model, input_signature=spec, opset=16)

# Save the ONNX model
with open(onnx_model_path, "wb") as f:
    f.write(onnx_model.SerializeToString())
```
