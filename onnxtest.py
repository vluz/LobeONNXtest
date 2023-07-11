import argparse
import json
import os
import numpy as np
import onnxruntime as rt
from PIL import Image


class ONNXModel:
    def __init__(self, dir_path) -> None:
        model_dir = dir_path
        with open(os.path.join(model_dir, "signature.json"), "r") as f:
            self.signature = json.load(f)
        self.model_file = os.path.join(model_dir, self.signature.get("filename"))
        self.signature_inputs = self.signature.get("inputs")
        self.signature_outputs = self.signature.get("outputs")
        self.session = None
        version = self.signature.get("export_model_version")


    def load(self) -> None:
        self.session = rt.InferenceSession(path_or_bytes=self.model_file)


    def predict(self, image: Image.Image) -> dict:
        img = self.process_image(image, self.signature_inputs.get("Image").get("shape"))
        fetches = [(key, value.get("name")) for key, value in self.signature_outputs.items()]
        feed = {self.signature_inputs.get("Image").get("name"): [img]}
        outputs = self.session.run(output_names=[name for (_, name) in fetches], input_feed=feed)
        return self.process_output(fetches, outputs)


    def process_image(self, image: Image.Image, input_shape: list) -> np.ndarray:
        width, height = image.size
        if image.mode != "RGB":
            image = image.convert("RGB")
        if width != height:
            square_size = min(width, height)
            left = (width - square_size) / 2
            top = (height - square_size) / 2
            right = (width + square_size) / 2
            bottom = (height + square_size) / 2
            image = image.crop((left, top, right, bottom))
        input_width, input_height = input_shape[1:3]
        if image.width != input_width or image.height != input_height:
            image = image.resize((input_width, input_height))
        image = np.asarray(image) / 255.0
        return image.astype(np.float32)


    def process_output(self, fetches: dict, outputs: dict) -> dict:
        out_keys = ["label", "confidence"]
        results = {}
        for i, (key, _) in enumerate(fetches):
            val = outputs[i].tolist()[0]
            if isinstance(val, bytes):
                val = val.decode()
            results[key] = val
        confs = results["Confidences"]
        labels = self.signature.get("classes").get("Label")
        output = [dict(zip(out_keys, group)) for group in zip(labels, confs)]
        sorted_output = {"predictions": sorted(output, key=lambda k: k["confidence"], reverse=True)}
        return sorted_output


if __name__ == "__main__":
    rt.set_default_logger_severity(3)
    parser = argparse.ArgumentParser(description="Predict a label for an image.")
    parser.add_argument("image", help="Path to your image file.")
    args = parser.parse_args()
    dir_path = os.getcwd()
    if os.path.isfile(args.image):
        image = Image.open(args.image)
        model = ONNXModel(dir_path=dir_path)
        model.load()
        outputs = model.predict(image)
        print(json.dumps(outputs, indent=4, sort_keys=False))
    else:
        print(f"Fine not foud: {args.image}")
