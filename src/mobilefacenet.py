"""
Module for feature extraction (embeddings) with MobileFaceNet model.
"""

import tensorflow as tf
import numpy as np
import logging
import cv2


class MobileFaceNet:
    def __init__(self, model_type="unquantized"):
        self.model_type = model_type

        # Initialize model based on model type
        self.input_details = None
        self.input_batch_size = None
        self.input_height = None # will be 112
        self.input_width = None # will be 112
        self.channels = None
        self.output_details = None
        self.initialize_model(model_type)

        # Initialize variables for inference
        self.img_height = None
        self.img_width = None
        self.img_channels = None

        logging.info("Initialized MobileFaceNet Extractor")

    def initialize_model(self, model_type):
        if model_type == "unquantized":
            self.interpreter = tf.lite.Interpreter(
                model_path="models/MobileFaceNet/MobileFaceNet_TF_prefrozen/mobilefacenet_unq_TF211.tflite"
            )
        elif model_type == "quantized":
            self.interpreter = tf.lite.Interpreter(
                model_path="models/MobileFaceNet/MobileFaceNet_TF_prefrozen/mobilefacenet_dyq_TF211.tflite"
            )
        elif model_type == "ente_web":
            self.interpreter = tf.lite.Interpreter(
                model_path="models/MobileFaceNet/mobilefacenet_ente_web.tflite"
            )
        self.interpreter.allocate_tensors()

        # Get model info
        self.get_model_input_details()
        self.get_model_output_details()

    def get_model_input_details(self):
        """Get model input details"""
        self.input_details = self.interpreter.get_input_details()
        input_shape = self.input_details[0]["shape"]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.channels = input_shape[3]

    def get_model_output_details(self):
        """Get model output details"""
        self.output_details = self.interpreter.get_output_details()
        output_shape = self.output_details[0]["shape"]
        self.output_batch_size = output_shape[0]
        self.output_embedding_size = output_shape[1]

    def extract_embedding(self, image, bounding_boxes=None, prewhiten=False):
        """Extract embedding from image"""
        # TODO: Add support for batch inference (is possible for this model!)
        # Prepare image for inference
        input_tensor = self.preprocess_input(image, bounding_boxes, prewhiten)

        # Perform inference on the image
        output = self.inference(input_tensor)

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

        return tensor

    def inference(self, input_tensor):
        """Perform inference on the image"""
        self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])

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
