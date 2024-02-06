"""
Module for feature extraction (embeddings) with MobileFaceNet model.
"""

import logging
import time
import tensorflow as tf
import numpy as np
import cv2
import onnx
import onnxruntime as ort


class MobileFaceNetOnnx:
    def __init__(self, model_type=15):
        self.model_type = model_type

        # Initialize model based on model type
        self.input_details = None
        self.input_batch_size = None
        self.input_height = 112
        self.input_width = 112
        self.output_details = None
        self.initialize_model(model_type)

        # Initialize variables for inference
        self.img_height = None
        self.img_width = None
        self.img_channels = None

        logging.info("Initialized MobileFaceNet Extractor")

    def initialize_model(self, model_type):
        if model_type == 15:
            self.model_path = "models/MobileFaceNet/onnx/mobilefacenet_opset15.onnx"
        elif model_type == 18:
            self.model_path = "models/MobileFaceNet/onnx/mobilefacenet_opset18.onnx"
        else:
            raise ValueError("Invalid model type! Use 15 or 18.")

        # Load model
        self.model = onnx.load(self.model_path)

        # Check model
        onnx.checker.check_model(self.model)

        # Print a human readable representation of the graph
        onnx.helper.printable_graph(self.model.graph)

        # Initialize the onnxruntime inference session
        self.session = ort.InferenceSession(self.model_path)

    def extract_embedding(
        self, image, bounding_boxes=None, prewhiten=False, print_time=False, return_time=False
    ):
        """Extract embedding from image"""
        # TODO: Add support for batch inference (is possible for this model!)
        # Prepare image for inference
        input_tensor = self.preprocess_input(image, bounding_boxes, prewhiten)

        # Perform inference on the image
        start_time = time.time()
        output = self.inference(input_tensor)
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # Convert to milliseconds

        if print_time:        
            print(f"Inference time: {inference_time} milliseconds")

        if return_time:
            return output, inference_time

        return output

    def preprocess_input(self, image_numpy, bounding_boxes=None, prewhiten=False):
        # img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image_numpy.ndim == 3:
            self.img_height, self.img_width, self.img_channels = image_numpy.shape
            self.input_batch_size = 1
        elif image_numpy.ndim == 4:
            (
                self.input_batch_size,
                self.img_height,
                self.img_width,
                self.img_channels,
            ) = image_numpy.shape
        else:
            raise ValueError(
                "Wrong input image dimension! Expected 3 (H,W,C) or 4 (H,W,C) dimensions."
            )

        if prewhiten:
            image_numpy = self._prewhiten(image_numpy)
        else:
            image_numpy = self._fixed_standardization(image_numpy)

        # Resize image to model input size
        img_resized = tf.image.resize(
            image_numpy,
            [self.input_height, self.input_width],
            method="bicubic",
            preserve_aspect_ratio=False,
        )

        # Convert image to tensor
        tensor = (
            tf.expand_dims(img_resized, axis=0)
            if len(img_resized.shape) == 3
            else img_resized
        )

        return tensor.numpy()

    def inference(self, input_tensor):
        """Perform inference on the image"""
        output = self.session.run(["embeddings"], {"img_inputs": input_tensor})[0][0]

        return output

    @staticmethod
    def _fixed_standardization(x):
        """Change value range from [0, 255] to [-1, 1]"""
        x = (np.float32(x) - 127.5) / 128.0
        return x

    @staticmethod
    def _prewhiten(x):
        """Preprocessing step for facenet model, specific per image"""
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)

        return y
