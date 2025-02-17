import tensorflow as tf
import numpy as np
import onnxruntime as ort
import cv2
import os


class SyndromeProcessing:
    def __init__(self, resolution=250):
        self.resolution = resolution
        self.valid_extensions = (".jpg", ".jpeg", ".png")

    def load_and_preprocess_image(self, image_path):
        img = cv2.imread(image_path)  # Load image using OpenCV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        img = cv2.resize(img, (self.resolution, self.resolution))  # Resize
        img = img.astype(np.float32) / 255.0  # Normalize to [0,1]
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        return img

    def find_images_in_dir(self, directory):
        return [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
            and f.lower().endswith(self.valid_extensions)
        ]

    def preprocess(self, dir_path):
        return [
            self.load_and_preprocess_image(path)
            for path in self.find_images_in_dir(dir_path)
        ]

    def decode_prediction(self, prediction):
        confidence = prediction if prediction > 0.5 else 1 - prediction
        pred = "down" if prediction > 0.5 else "healthy"
        return {"prediction": pred, "confidence": float(confidence)}

    def postprocess(self, outputs):
        return [self.decode_prediction(out[0][0]) for out in outputs]


class SyndromeDetectionModel:
    def __init__(self, model_path):
        self.processor = SyndromeProcessing()
        self.session = ort.InferenceSession(
            model_path,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def predict(self, image_path):
        input_img = self.processor.load_and_preprocess_image(image_path)
        output = self.session.run(None, {"input": input_img})
        result = self.processor.decode_prediction(output[0])
        result["image_path"] = image_path
        return result

    def predict_dir(self, input_dir):
        outputs = []
        for inp in self.processor.find_images_in_dir(input_dir):
            outputs.append(self.predict(inp))
        return outputs
