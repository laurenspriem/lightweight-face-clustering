import time
import cv2
import numpy as np
import tensorflow as tf
from src.blazeface_utils import gen_anchors, SsdAnchorsCalculatorOptions

# pylint: disable=C0116, E0401, E1101


class BlazeFace:
    """BlazeFace model class for face detection"""

    def __init__(
        self,
        model_type="sparse",
        score_threshold=0.6,
        iou_threshold=0.3,
        max_face_num=100,
        key_point_size=6,
    ):
        self.type = model_type
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_face_num = max_face_num
        self.key_point_size = key_point_size
        self.inverse_sigmoid_score_threshold = np.log(
            self.score_threshold / (1 - self.score_threshold)
        )
        self.fps = 0
        self.time_initialized = time.time()
        self.time_last_prediction = None
        # self.base_path = os.getcwd()
        self.frame_counter = 0

        # Initialize model based on model type
        self.input_details = None
        self.input_batch_size = None
        self.input_height = None
        self.input_width = None
        self.channels = None
        self.output_details = None
        self.initialize_model(model_type)

        # Generate anchors for model
        self.generate_anchors(model_type)

        # Initialize variables for inference
        self.img_height = None
        self.img_width = None
        self.img_channels = None

    def initialize_model(self, model_type):
        if model_type == "sparse":
            self.interpreter = tf.lite.Interpreter(
                model_path="models/BlazeFace/Full-range-sparse/face_detection_full_range_sparse.tflite"
            )
        elif model_type == "dense":
            self.interpreter = tf.lite.Interpreter(
                model_path="models/BlazeFace/Full-range-dense/face_detection_full_range.tflite"
            )
        self.interpreter.allocate_tensors()

        # Get model info
        self.get_model_input_details()
        self.get_model_output_details()

    def detect_faces(self, image):
        # Prepare image for inference
        input_tensor = self.preprocess_input(image)

        # Perform inference on the image
        output0, output1 = self.inference(input_tensor)

        # Filter scores based on the detection scores
        scores, good_detection_indices = self.filter_detections(output1)

        # Extract information of filtered detections
        boxes, keypoints = self.extract_detections(output0, good_detection_indices)

        # Filter results with non-maximum suppression
        detection_results = self.filter_with_non_max_supression(
            boxes, scores, keypoints
        )

        return detection_results

    def draw_detections_single_image(self, img, results):
        # take copy of image to prevent overwriting original image
        img = img.copy()

        bounding_boxes = results.boxes
        keypoints = results.keypoints
        scores = results.scores

        # Add bounding boxes, scores and keypoints
        for bounding_box, keypoints, score in zip(bounding_boxes, keypoints, scores):
            # bounding boxes
            x1 = (self.img_width * bounding_box[0]).astype(int)
            x2 = (self.img_width * bounding_box[2]).astype(int)
            y1 = (self.img_height * bounding_box[1]).astype(int)
            y2 = (self.img_height * bounding_box[3]).astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (250, 0, 0), 2)

            # scores
            cv2.putText(
                img,
                "{:.2f}".format(score),
                (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (250, 0, 0),
                2,
            )

            # Add keypoints for the current face
            for keypoint in keypoints:
                xKeypoint = (keypoint[0] * self.img_width).astype(int)
                yKeypoint = (keypoint[1] * self.img_height).astype(int)
                cv2.circle(img, (xKeypoint, yKeypoint), 4, (22, 22, 250), -1)

        return img

    def get_model_input_details(self):
        """Get model input details"""
        self.input_details = self.interpreter.get_input_details()
        input_shape = self.input_details[0]["shape"]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.channels = input_shape[3]

    def get_model_output_details(self):
        self.output_details = self.interpreter.get_output_details()

    def generate_anchors(self, model_type):
        if model_type == "sparse":
            # Options to generate anchors for SSD object detection models.
            ssd_anchors_calculator_options = SsdAnchorsCalculatorOptions(
                input_size_width=192,
                input_size_height=192,
                min_scale=0.1484375,
                max_scale=0.75,
                num_layers=1,  # 4
                feature_map_width=[],
                feature_map_height=[],
                strides=[4],
                aspect_ratios=[1.0],
                anchor_offset_x=0.5,
                anchor_offset_y=0.5,
                reduce_boxes_in_lowest_layer=False,
                interpolated_scale_aspect_ratio=0.0,
                fixed_anchor_size=True,
            )

        elif model_type == "dense":
            # Options to generate anchors for SSD object detection models.
            ssd_anchors_calculator_options = SsdAnchorsCalculatorOptions(
                input_size_width=192,
                input_size_height=192,
                min_scale=0.1484375,  # 0.15625,
                max_scale=0.75,
                num_layers=1,  # 4,
                feature_map_width=[],
                feature_map_height=[],
                strides=[4],  # [16, 32, 32, 32],
                aspect_ratios=[1.0],
                anchor_offset_x=0.5,
                anchor_offset_y=0.5,
                reduce_boxes_in_lowest_layer=False,
                interpolated_scale_aspect_ratio=0.0,
                fixed_anchor_size=True,
            )

        self.anchors = gen_anchors(ssd_anchors_calculator_options)

    def preprocess_input(self, image_numpy):
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

        # Input values should be from -1 to 1 with a size of 160 x 192 pixels
        image_numpy = (image_numpy / 127.5) - 1.0

        img_resized = tf.image.resize(
            image_numpy,
            [self.input_height, self.input_width],
            method="bicubic",
            preserve_aspect_ratio=False,
        )

        tensor = (
            tf.expand_dims(img_resized, axis=0)
            if len(img_resized.shape) == 3
            else img_resized
        )

        return tensor

    def inference(self, input_tensor):
        # TODO: make the function work with batches (resize_input_tensor)

        # Resize the input tensor if the batch size is greater than 1
        if self.input_batch_size == 1:
            self.interpreter.set_tensor(self.input_details[0]["index"], input_tensor)
            self.interpreter.invoke()

            # Matrix of 2304 x 16 with information about the detected faces (bounding boxes and landmarks)
            output0 = np.squeeze(
                self.interpreter.get_tensor(self.output_details[0]["index"])
            )

            # Matrix of 2304 with the raw detection scores for the detected faces
            output1 = np.squeeze(
                self.interpreter.get_tensor(self.output_details[1]["index"])
            )
        elif self.input_batch_size > 1:
            output0 = np.zeros((self.input_batch_size, 2304, 16))
            output1 = np.zeros((self.input_batch_size, 2304))
            # individual_input_tensor = tf.zeros(
            #     (1, self.input_height, self.input_width, self.channels)
            # )

            for i in range(self.input_batch_size):
                individual_input_tensor = tf.expand_dims(input_tensor[i], 0)
                self.interpreter.set_tensor(
                    self.input_details[0]["index"], individual_input_tensor
                )
                self.interpreter.invoke()

                # Matrix of 2304 x 16 with information about the detected faces (bounding boxes and landmarks)
                output0[i] = np.squeeze(
                    self.interpreter.get_tensor(self.output_details[0]["index"])
                )

                # Matrix of 2304 with the raw detection scores for the detected faces
                output1[i] = np.squeeze(
                    self.interpreter.get_tensor(self.output_details[1]["index"])
                )

        return output0, output1

    def filter_detections(self, output1):
        if self.input_batch_size == 1:
            # Filter based on the score threshold before applying sigmoid function
            good_detection_indices = np.flatnonzero(
                output1 > self.inverse_sigmoid_score_threshold
            )
            # Convert scores to sigmoid values between 0 and 1
            scores = 1.0 / (1.0 + np.exp(-output1[good_detection_indices]))
        else:
            # Find the good detection indices
            good_detection_indices = np.argwhere(
                output1 > self.inverse_sigmoid_score_threshold
            )
            # Extract the good detection scores from output1 using the indices
            scores = output1[good_detection_indices[:, 0], good_detection_indices[:, 1]]
            # Apply the sigmoid function to the extracted scores
            scores = 1.0 / (1.0 + np.exp(-scores))

        return scores, good_detection_indices

    def extract_detections(self, output0, good_detection_indices):
        if self.input_batch_size != 1:
            raise ValueError("Batch size greater than 1 not supported yet!")

        num_good_detections = good_detection_indices.shape[0]

        keypoints = np.zeros((num_good_detections, self.key_point_size, 2))
        boxes = np.zeros((num_good_detections, 4))
        for idx, detection_idx in enumerate(good_detection_indices):
            anchor = self.anchors[detection_idx]

            sx = output0[detection_idx, 0]
            sy = output0[detection_idx, 1]
            w = output0[detection_idx, 2]
            h = output0[detection_idx, 3]

            cx = sx + anchor.x_center * self.input_width  # width and height are 192
            cy = sy + anchor.y_center * self.input_height

            cx /= self.input_width
            cy /= self.input_height
            w /= self.input_width
            h /= self.input_height

            for j in range(self.key_point_size):
                lx = output0[detection_idx, 4 + (2 * j) + 0]
                ly = output0[detection_idx, 4 + (2 * j) + 1]
                lx += anchor.x_center * self.input_width
                ly += anchor.y_center * self.input_height
                lx /= self.input_width
                ly /= self.input_height
                keypoints[idx, j, :] = np.array([lx, ly])

            boxes[idx, :] = np.array(
                [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5]
            )  # boxes gives the bounding box coordinates: [x1, y1, x2, y2]

        return boxes, keypoints

    def filter_with_non_max_supression(self, boxes, scores, keypoints):
        # TODO - weighted non_max_suppression as in BlazeFace paper
        selected_indices = tf.image.non_max_suppression(
            boxes, scores, self.max_face_num, self.iou_threshold
        )
        filtered_boxes = tf.gather(boxes, selected_indices).numpy()
        filtered_keypoints = tf.gather(keypoints, selected_indices).numpy()
        filtered_scores = tf.gather(scores, selected_indices).numpy()

        detection_results = BlazeFaceResults(
            filtered_boxes, filtered_scores, filtered_keypoints
        )
        return detection_results


class BlazeFaceResults:
    def __init__(self, boxes, scores, keypoints):
        self.boxes = boxes
        self.keypoints = keypoints
        self.scores = scores

    def __repr__(self):
        return (
            f"BlazeFaceResults(\n"
            f"  boxes={self.boxes}  # [xmin, ymin, xmax, ymax] in relative terms [0,1]\n"
            f"  scores={self.scores}  # Detection scores\n"
            f"  keypoints={self.keypoints}  # [x, y] coordinates for each (of 6) keypoints (left eye, right eye, nose tip, mouth, left eye tragion, right eye tragion)\n"
            f")"
        )
