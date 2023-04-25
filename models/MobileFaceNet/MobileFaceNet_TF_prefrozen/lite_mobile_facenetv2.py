# -*- coding: utf-8 -*-

"""
Module for descriptors extraction with FaceNet model.
"""

import tensorflow as tf
import numpy as np
import logging
import cv2


class LiteMobileFacenetExtractor:
    """
    Descriptors extractor with MobileFaceNet model.

    Attributes
    ----------
    _interpreter: tf.Lite.Interpreter
        TFLite inference interpreter.

    """

    def __init__(self, model):
        """
        Initialize MobileFacenet extractor with loaded facenet model.

        Parameters
        ----------
        model: str
            Path to model (.tflite file).

        """
        # Load TFLite model and allocate tensors.
        self._interpreter = tf.lite.Interpreter(model_path=model)
        self._interpreter.allocate_tensors()

        # Get input and output tensors.
        self._input_details = self._interpreter.get_input_details()
        self._output_details = self._interpreter.get_output_details()
        logging.info('Init Lite MobileFaceNet Extractor')

    def extract(self, image, bboxes):
        """
        Extract descriptor for every person on image.

        Parameters
        ----------
        image: numpy array
            BGR image.
        bboxes: numpy array
            Non empty array with several bounding boxes.
            Format is [x, y, width, height] where each value is int.

        Returns
        -------
        numpy array
            Array of descriptors with shape (len(bboxes), descriptor_size),
            descriptor_size depends on MobileFacenet model (128)
        """
        assert (len(bboxes) > 0)
        assert (image.shape[2] == 3)
        from copy import copy

        clipped_bboxes = []
        for bbox in bboxes:
            clipped_bboxes.append(copy(bbox))
        for bbox in clipped_bboxes:
            if bbox[0] < 0:
                bbox[0] = 0
            if bbox[1] < 0:
                bbox[1] = 0
            if bbox[2] > image.shape[0]:
                bbox[2] = image.shape[0]
            if bbox[3] > image.shape[1]:
                bbox[3] = image.shape[1]

        input_images = self._prepare_input(image, clipped_bboxes)
        input_images = input_images.astype('float32')
        self._interpreter.set_tensor(self._input_details[0]['index'],
                                     input_images)

        self._interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        emb = self._interpreter.get_tensor(self._output_details[0]['index'])

        return emb

    def _prepare_input(self, image, bboxes, image_size=112):
        """
        For each bounding box crop image, resize it and whiten.

        Parameters
        ----------
        image: numpy array
            BGR image.
        bboxes: numpy array
            Array with several bounding boxes.
            Format is [x, y, width, height] where each value is int.
        image_size: int
            Size of image in model input.

        Returns
        -------
        numpy array
            Tensor with images of size (len_bboxes, image_size, image_size, 3)
        """
        assert (len(bboxes) > 0)
        assert (image.shape[2] == 3)
        # Convert BGR to RGB
        image = image[:, :, ::-1]

        faces = []
        for bbox in bboxes:
            cropped = image[bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2], :]
            aligned = cv2.resize(cropped, (image_size, image_size),
                                 interpolation=cv2.INTER_LINEAR)
            prewhitened = self._prewhiten(aligned)
            faces.append(prewhitened)
        return np.stack(faces)

    def _prewhiten(self, x):
        """
        Whiten image (with mean and std).

        Parameters
        ----------
        x: numpy array
            RGB image.

        Returns
        -------
        numpy array
            Whitened image.
        """
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        return y

    def release(self):
        """
        Release object and close TFSession.

        """
        pass
