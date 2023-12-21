import time
import cv2
import numpy as np
import onnx
import onnxruntime as ort

# pylint: disable=C0116, E0401, E1101


class YoloFace:
    """YOLOv5Face model class for face detection"""

    def __init__(
        self,
        score_threshold=0.8,
        iou_threshold=0.4,
        max_face_num=100,
        key_point_size=5,
    ):
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.max_face_num = max_face_num
        self.key_point_size = key_point_size
        self.inverse_sigmoid_score_threshold = np.log(
            self.score_threshold / (1 - self.score_threshold)
        )
        # self.base_path = os.getcwd()

        # Initialize model
        self.input_height = 640
        self.input_width = 640
        self.channels = 3

        self.initialize_model()

    def initialize_model(self):
        # Load model
        self.model_path = "models/YOLOv5Face/yolov5s_face_640_640_dynamic.onnx"
        self.model = onnx.load(self.model_path)

        # Check model
        onnx.checker.check_model(self.model)

        # Print a human readable representation of the graph
        onnx.helper.printable_graph(self.model.graph)

        # Initialize the onnxruntime inference session
        self.session = ort.InferenceSession(self.model_path)

    def detect_faces(self, image):
        # preprocessing image
        # Input values should be from 0 to 1 with a size of 640 x 640 pixels
        image_numpy = image / 255
        # Resize the image to [b, 3, 640, 640] using cv2
        img_resized = cv2.resize(
            image_numpy,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        img_resized = np.transpose(img_resized, (2, 0, 1))
        img_resized = np.expand_dims(img_resized, axis=0)
        img_resized = img_resized.astype(np.float32)

        # Perform inference on the image
        output = self.session.run(["output"], {'input': img_resized})[0][0] # ndarray (25200, 16)

        # Retrieve the boxes and scores from the YOLOv5Face output
        boxes = output[:, :4]
        scores = output[:, 4]

        # Set the score threshold and NMS threshold
        score_threshold = 0.5
        nms_threshold = 0.5

        # Apply non-maximum suppression to filter the detections
        indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, nms_threshold)
        filtered_output = output[indices]

        boxes = filtered_output[:, :4] / 640
        boxes[:, 0] = boxes[:, 0] - boxes[:, 2]/2
        boxes[:, 1] = boxes[:, 1] - boxes[:, 3]/2
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        scores = filtered_output[:, 4]

        keypoints = filtered_output[:, 5:15] / 640
        # reshape from [:, 10] to [:, 5, 2]
        keypoints = keypoints.reshape(keypoints.shape[0], 5, 2)

        detection_results = YOLOFaceResults(
            boxes, scores, keypoints
        )

        return detection_results

    def draw_detections_single_image(self, img, results):
        # take copy of image to prevent overwriting original image
        img = img.copy()
        img_height, img_width, img_channels = img.shape

        bounding_boxes = results.boxes
        keypoints = results.keypoints
        scores = results.scores

        # Add bounding boxes, scores and keypoints
        for bounding_box, keypoints, score in zip(bounding_boxes, keypoints, scores):
            # bounding boxes
            x1 = (img_width * bounding_box[0]).astype(int)
            x2 = (img_width * bounding_box[2]).astype(int)
            y1 = (img_height * bounding_box[1]).astype(int)
            y2 = (img_height * bounding_box[3]).astype(int)
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
                xKeypoint = (keypoint[0] * img_width).astype(int)
                yKeypoint = (keypoint[1] * img_height).astype(int)
                cv2.circle(img, (xKeypoint, yKeypoint), 4, (22, 22, 250), -1)

        return img


class YOLOFaceResults:
    def __init__(self, boxes, scores, keypoints):
        self.boxes = boxes
        self.keypoints = keypoints
        self.scores = scores

    def __repr__(self):
        return (
            f"YOLOFaceResults(\n"
            f"  boxes={self.boxes}  # [xmin, ymin, xmax, ymax] in relative terms [0,1]\n"
            f"  scores={self.scores}  # Detection scores\n"
            f"  keypoints={self.keypoints}  # [x, y] coordinates for each (of 6) keypoints (left eye, right eye, nose tip, mouth, left eye tragion, right eye tragion)\n"
            f")"
        )

    def get_face_count(self):
        return len(self.boxes)

    def get_largest_face(self):
        if len(self.boxes) == 0:
            return None  # No faces detected

        max_area = 0
        largest_face_index = -1

        for i, box in enumerate(self.boxes):
            xmin, ymin, xmax, ymax = box
            area = (xmax - xmin) * (ymax - ymin)

            if area > max_area:
                max_area = area
                largest_face_index = i

        if largest_face_index == -1:
            return (
                None  # No largest face found, should not happen unless boxes is empty
            )

        return YOLOFaceResults(
            boxes=np.array([self.boxes[largest_face_index]]),
            scores=np.array([self.scores[largest_face_index]]),
            keypoints=np.array([self.keypoints[largest_face_index]]),
        )
