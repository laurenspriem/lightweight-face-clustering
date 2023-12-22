import math
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
from scipy.ndimage import rotate
from skimage.transform import SimilarityTransform


class ArcFaceAlignment:
    def __init__(self, landmarks=4):
        self.landmarks = landmarks
        if landmarks == 4:
            self.arcface_src = np.expand_dims(
                np.array(
                    [
                        [38.2946, 51.6963],
                        [73.5318, 51.5014],
                        [56.0252, 71.7366],
                        [56.1396, 92.2848],
                    ],
                    dtype=np.float32,
                ),
                axis=0,
            )
        if landmarks == 5:
            self.arcface_src = np.expand_dims(
                np.array(
                    [
                        [38.2946, 51.6963],
                        [73.5318, 51.5014],
                        [56.0252, 71.7366],
                        [41.5493, 92.3655],
                        [70.7299, 92.2041],
                    ],
                    dtype=np.float32,
                ),
                axis=0,
            )

    def crop_and_align(self, image, face_detection_result):
        if image.ndim == 4:
            image = image[0]
        h, w = image.shape[:2]

        warped_faces = []
        for keypoints in face_detection_result.keypoints:
            # Get the absolute face landmarks
            landmarks_absolute_full = keypoints[: self.landmarks]
            landmarks_absolute_full[:, 0] *= w
            landmarks_absolute_full[:, 1] *= h
            landmarks_absolute_full = landmarks_absolute_full.astype(np.int32)

            # Perform the alignment via transformation
            warped_face = self.norm_crop(image, landmarks_absolute_full)
            warped_faces.append(warped_face)

        warped_face_results = AlignedFaceResults(
            aligned_faces=warped_faces,
            num_faces=len(warped_faces),
            eye_coordinates=None,
            rotation_angles=None,
            relative_face_sizes=[
                (face.shape[1] * face.shape[0]) / (h * w) for face in warped_faces
            ],
        )

        return warped_face_results

    def estimate_norm(self, lmk, image_size=112, mode="arcface"):
        # lmk is prediction; src is template
        assert lmk.shape == (self.landmarks, 2)
        tform = SimilarityTransform()
        lmk_tran = np.insert(lmk, 2, values=np.ones(self.landmarks), axis=1)
        min_M = []
        min_index = []
        min_error = float("inf")
        if mode == "arcface":
            assert image_size == 112
            src = self.arcface_src
        else:
            raise (
                ValueError("Other modes than for ArcFace not supported at this time")
            )
            # src = src_map[image_size]
        for i in np.arange(src.shape[0]):
            tform.estimate(lmk, src[i])
            M = tform.params[0:2, :]
            results = np.dot(M, lmk_tran.T)
            results = results.T
            error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
            if error < min_error:
                min_error = error
                min_M = M
                min_index = i
        return min_M, min_index

    def norm_crop(self, img, landmark, image_size=112, mode="arcface"):
        # landmark has to be absolute landmarks in integers!!!!
        M, pose_index = self.estimate_norm(landmark, image_size, mode)
        img_shape = img.shape
        warped = cv2.warpAffine(
            img, M, (image_size, image_size), borderValue=0.0, flags=cv2.INTER_CUBIC
        )
        return warped


class SimpleFaceAlignment:
    def __init__(self):
        pass

    def crop_and_align(self, image, face_detection_result):
        if image.ndim == 4:
            image = image[0]

        h, w = image.shape[:2]
        cropped_faces, xmin, ymin = self.crop(image, face_detection_result, h, w)
        aligned_faces_results = self.align(
            cropped_faces, xmin, ymin, face_detection_result, h, w
        )
        return aligned_faces_results

    def crop(self, image, face_detection_result, h, w):
        cropped_faces, xmin, ymin = [], [], []

        for box in face_detection_result.boxes:
            xmin.append(max(0, (box[0] * w).astype(int)))
            ymin.append(max(0, (box[1] * h).astype(int)))
            xmax = min(w, (box[2] * w).astype(int))
            ymax = min(h, (box[3] * h).astype(int))
            cropped_faces.append(image[ymin[-1] : ymax, xmin[-1] : xmax])

        return cropped_faces, xmin, ymin

    def align(self, cropped_faces, xmin, ymin, face_detection_result, h, w):
        eye1, eye2, angles, aligned_face_images = [], [], [], []
        for idx, landmarks in enumerate(face_detection_result.keypoints):
            eye1.append(
                (
                    (landmarks[0][0] * w).astype(int) - xmin[idx],
                    (landmarks[0][1] * h).astype(int) - ymin[idx],
                )
            )
            eye2.append(
                (
                    (landmarks[1][0] * w).astype(int) - xmin[idx],
                    (landmarks[1][1] * h).astype(int) - ymin[idx],
                )
            )
            third_point, direction = self.create_third_point(eye1[-1], eye2[-1])

            if direction:
                a = self.euclidean_distance(eye1[-1], third_point)
                b = self.euclidean_distance(eye2[-1], eye1[-1])
                c = self.euclidean_distance(eye2[-1], third_point)

                cos_a = (b * b + c * c - a * a) / (2 * b * c)
                angle = np.arccos(cos_a)
                angle = (angle * 180) / math.pi
                if direction == 1:
                    angle = 90 - angle
                angles.append(angle)

                cropped_face_image = Image.fromarray(cropped_faces[idx])
                aligned_face_images.append(
                    np.array(
                        cropped_face_image.rotate(
                            -direction * angle,
                            resample=Image.BICUBIC,
                            fillcolor=(0, 0, 0),
                        )
                    )
                )
            else:
                angles.append(0)
                aligned_face_images.append(cropped_faces[idx])

        aligned_face_results = AlignedFaceResults(
            aligned_faces=aligned_face_images,
            num_faces=len(aligned_face_images),
            eye_coordinates=list(zip(eye1, eye2)),
            rotation_angles=angles,
            relative_face_sizes=[
                (face.shape[1] * face.shape[0]) / (h * w)
                for face in aligned_face_images
            ],
        )

        return aligned_face_results

    @staticmethod
    def create_third_point(point1, point2):
        if point1[1] < point2[1]:
            return (point1[0], point2[1]), -1
        elif point1[1] > point2[1]:
            return (point2[0], point1[1]), 1
        else:
            return None, 0

    @staticmethod
    def euclidean_distance(a, b):
        x1 = a[0]
        y1 = a[1]
        x2 = b[0]
        y2 = b[1]
        return math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))


class FaceMeshAlignment:
    def __init__(self, model_type="regular", crop=True):
        self.crop = crop
        self.left_eye_iris_indices = [468, 469, 470, 471, 472]
        self.right_eye_iris_indices = [473, 474, 475, 476, 477]
        self.face_oval_indices = [
            10,
            338,
            338,
            297,
            297,
            332,
            332,
            284,
            284,
            251,
            251,
            389,
            389,
            356,
            356,
            454,
            454,
            323,
            323,
            361,
            361,
            288,
            288,
            397,
            397,
            365,
            365,
            379,
            379,
            378,
            378,
            400,
            400,
            377,
            377,
            152,
            152,
            148,
            148,
            176,
            176,
            149,
            149,
            150,
            150,
            136,
            136,
            172,
            172,
            58,
            58,
            132,
            132,
            93,
            93,
            234,
            234,
            127,
            127,
            162,
            162,
            21,
            21,
            54,
            54,
            103,
            103,
            67,
            67,
            109,
            109,
            10,
        ]
        self.face_oval_indices_zipped = list(
            zip(self.face_oval_indices[0::2], self.face_oval_indices[1::2])
        )

        self.initialize_model(model_type)

    def initialize_model(self, model_type):
        if model_type == "regular":
            self.interpreter = tf.lite.Interpreter(
                model_path="models/Face-Mesh/regular/face_landmark.tflite"
            )
        elif model_type == "attention":
            self.interpreter = tf.lite.Interpreter(
                model_path="models/Face-Mesh/attention/face_landmark_with_attention.tflite"
            )
        self.interpreter.allocate_tensors()

        # Get model input details
        self.input_details = self.interpreter.get_input_details()
        input_shape = self.input_details[0]["shape"]
        self.input_height = input_shape[1]
        self.input_width = input_shape[2]
        self.channels = input_shape[3]

        # Get model output details
        self.output_details = self.interpreter.get_output_details()
        self.landmarks_num = self.output_details[0]["shape"][3] // 3

    def normalize_faces(self, image, face_detection_result):
        if image.ndim == 4:
            image = image[0]
        h, w = image.shape[:2]

        # Get the face mesh
        face_meshes, xmin, ymin = self.get_face_meshes(
            image, face_detection_result, h, w
        )

        # Rotate the face mesh
        rotated_face_meshes, eye1, eye2, angles = self.rotate_face_meshes(
            face_meshes, xmin, ymin, face_detection_result, h, w
        )
        # Crop the face mesh
        if self.crop:
            normalized_face_meshes = self.crop_face_meshes(rotated_face_meshes)
        else:
            normalized_face_meshes = rotated_face_meshes

        aligned_face_results = AlignedFaceResults(
            aligned_faces=normalized_face_meshes,
            num_faces=len(normalized_face_meshes),
            eye_coordinates=list(zip(eye1, eye2)),
            rotation_angles=angles,
            relative_face_sizes=[
                (face.shape[1] * face.shape[0]) / (h * w)
                for face in normalized_face_meshes
            ],
        )
        return aligned_face_results

    def preprocess(self, image, face_detection_result, h, w, face_idx=0):
        xmin, ymin, xmax, ymax = face_detection_result.boxes[face_idx]
        w_relative, h_relative = abs(xmax - xmin), abs(ymax - ymin)
        xmin = max(0, xmin - 0.25 * w_relative)
        ymin = max(0, ymin - 0.25 * h_relative)
        xmax = min(1, xmax + 0.25 * w_relative)
        ymax = min(1, ymax + 0.25 * h_relative)

        image_processed = image[
            int(ymin * h) : int(ymax * h), int(xmin * w) : int(xmax * w)
        ]
        image_processed = (image_processed / 127.5) - 1.0
        image_processed = tf.image.resize(
            image_processed, (self.input_height, self.input_width), method="bicubic"
        )
        image_processed = (
            tf.expand_dims(image_processed, axis=0)
            if len(image_processed.shape) == 3
            else image_processed
        )

        return image_processed

    def inference(self, image):
        self.interpreter.set_tensor(self.input_details[0]["index"], image)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])
        output = output.reshape(self.landmarks_num, 3)[
            :, 0:2
        ]  # reshape into landmarks and take only x and z coordinates

        return output

    def get_face_meshes(
        self,
        original_image,
        face_detection_result,
        h,
        w,
        face_idx=0,
    ):
        face_meshes, xmin, ymin = [], [], []

        for face_idx in range(len(face_detection_result.boxes)):
            xmin.append(
                max(0, (face_detection_result.boxes[face_idx][0] * w).astype(int))
            )
            ymin.append(
                max(0, (face_detection_result.boxes[face_idx][1] * h).astype(int))
            )

            preprocessed_image = self.preprocess(
                original_image, face_detection_result, h, w, face_idx
            )
            inference_output = self.inference(preprocessed_image)

            factor = 1
            h_bias = abs(h - self.input_height) / 2
            w_bias = abs(w - self.input_width) / 2
            routes = []

            for source_idx, target_idx in self.face_oval_indices_zipped:
                source_point = inference_output[source_idx]
                target_point = inference_output[target_idx]

                relative_source = (
                    int(source_point[0] * factor + h_bias),
                    int(source_point[1] * factor + w_bias),
                )
                relative_target = (
                    int(target_point[0] * factor + h_bias),
                    int(target_point[1] * factor + w_bias),
                )
                routes.append(relative_source)
                routes.append(relative_target)

            mask = np.zeros((h, w, 3))
            mask = cv2.fillConvexPoly(mask, np.array(routes), (1, 1, 1))
            mask = mask.astype(bool)

            facemesh = np.zeros_like(original_image)
            facemesh[mask] = original_image[mask]
            face_meshes.append(facemesh)

        return face_meshes, xmin, ymin

    def rotate_face_meshes(self, face_meshes, xmin, ymin, face_detection_result, h, w):
        eye1, eye2, angles, rotated_face_meshes = [], [], [], []
        for idx, landmarks in enumerate(face_detection_result.keypoints):
            eye1.append(
                (
                    (landmarks[0][0] * w).astype(int) - xmin[idx],
                    (landmarks[0][1] * h).astype(int) - ymin[idx],
                )
            )
            eye2.append(
                (
                    (landmarks[1][0] * w).astype(int) - xmin[idx],
                    (landmarks[1][1] * h).astype(int) - ymin[idx],
                )
            )
            third_point, direction = self.create_third_point(eye1[-1], eye2[-1])

            if direction:
                a = self.euclidean_distance(eye1[-1], third_point)
                b = self.euclidean_distance(eye2[-1], eye1[-1])
                c = self.euclidean_distance(eye2[-1], third_point)

                cos_a = (b * b + c * c - a * a) / (2 * b * c)
                angle = np.arccos(cos_a)
                angle = (angle * 180) / math.pi
                if direction == 1:
                    angle = 90 - angle
                angles.append(angle)

                rotated_face_mesh = Image.fromarray(face_meshes[idx])
                rotated_face_meshes.append(
                    np.array(
                        rotated_face_mesh.rotate(
                            -direction * angle,
                            resample=Image.BICUBIC,
                            fillcolor=(0, 0, 0),
                        )
                    )
                )
            else:
                angles.append(0)
                rotated_face_meshes.append(face_meshes[idx])

        return rotated_face_meshes, eye1, eye2, angles

    @staticmethod
    def crop_face_meshes(face_meshes):
        cropped_face_meshes = []
        for face_mesh in face_meshes:
            # Create a binary mask where non-zero pixels are set to 1
            mask = np.any(face_mesh != 0, axis=-1)

            # Find the coordinates of the non-zero pixels
            coords = np.argwhere(mask)

            # Calculate the minimum and maximum coordinates in each axis (x, y)
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            # Crop the original image using the calculated coordinates
            cropped_image = face_mesh[y_min : y_max + 1, x_min : x_max + 1]

            cropped_face_meshes.append(cropped_image)

        return cropped_face_meshes

    @staticmethod
    def create_third_point(point1, point2):
        if point1[1] < point2[1]:
            return (point1[0], point2[1]), -1
        elif point1[1] > point2[1]:
            return (point2[0], point1[1]), 1
        else:
            return None, 0

    @staticmethod
    def euclidean_distance(a, b):
        x1 = a[0]
        y1 = a[1]
        x2 = b[0]
        y2 = b[1]
        return math.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))


class AlignedFaceResults:
    def __init__(
        self,
        aligned_faces,
        num_faces,
        eye_coordinates,
        rotation_angles,
        relative_face_sizes,
    ):
        self.aligned_faces = aligned_faces
        self.num_faces = num_faces
        self.eye_coordinates = eye_coordinates
        self.rotation_angles = rotation_angles
        self.relative_face_sizes = relative_face_sizes

    def __repr__(self):
        return (
            f"AlignedFaceResults(\n"
            f"  num_faces={self.num_faces},  # Number of detected faces\n"
            f"  eye_coordinates={self.eye_coordinates},  # (Left eye, Right eye)(from viewer perspective) coordinates for each face (x,y)\n"
            f"  rotation_angles={self.rotation_angles},  # Rotation angles for each face alignment\n"
            f"  relative_face_sizes={self.relative_face_sizes}  # Relative size of each face compared to the original image\n"
            ")"
        )
