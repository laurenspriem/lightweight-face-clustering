# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import cv2

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tqdm import tqdm
import time

# %%
path = os.getcwd()
path

# %%
# To speed up the notebook, set this value to True to skip intermediate demo steps and only run full pipeline at the end of this notebook
skip_intermediate_demo = True

# %% [markdown]
# ## Load data
#
# LFW subset locally downloaded

# %%
# Path to the directory containing the subset of LFW dataset
# dataset = 'lfw_10classes_20images'
dataset = "lfw_30classes_20images"
# dataset = 'lfw_30classes_2images'
# dataset = 'lfw_10images'
# dataset = 'lfw_original' # crashes the kernel, try in Paperspace!
data_path = path + "/data/" + dataset + "/"

# %%
# Load the subset data into a NumPy array
data = []
original_images = []
targets = []
targets = []
for subdir in os.listdir(data_path):
    subdir_path = os.path.join(data_path, subdir)
    if os.path.isdir(subdir_path):
        for file in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, file)
            if os.path.isfile(file_path):
                img = cv2.imread(file_path, cv2.IMREAD_COLOR)
                RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                data.append(RGB_img.flatten())
                original_images.append(RGB_img)
                targets.append(subdir)

data = np.array(data)
original_images = np.array(original_images)

# %% [markdown]
# ### Explore data

# %%


# %%
original_images.shape

# %%
# introspect the images arrays to find the shapes (for plotting)
n, w, h, c = original_images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = data
n_features = X.shape[1]

# the label to predict is the id of the person
target_names = targets
target_names_unique = [target_name for target_name in set(target_names)]
y = LabelEncoder().fit_transform(target_names.copy())
target_names_first_index = [np.where(y == label)[0][0] for label in np.unique(y)]
target_names_first_index.sort()

n_classes = len(target_names_unique)

subdir_counts = {subdir: targets.count(subdir) for subdir in set(targets)}
n_classes_larger_than_5 = len([count for count in subdir_counts.values() if count >= 5])
n_classes_larger_than_10 = len(
    [count for count in subdir_counts.values() if count >= 10]
)
n_images_from_classes_smaller_than_5 = sum(
    [count for count in subdir_counts.values() if count < 5]
)
n_images_from_classes_smaller_than_10 = sum(
    [count for count in subdir_counts.values() if count < 10]
)

# %%
# Print the shape of the data array
print("Data shape:", data.shape)

# Print the list of subdirectories and the number of images per subdirectory
print("Subdirectory counts (unordered):", subdir_counts, "\n")

print(
    f"There are {n} images in the dataset, {13233 - n} less than the original dataset (of 13233)."
)
print(f"Each image is {w} pixels wide and {h} pixels tall, with {c} channels (RGB).")
print(f"In total this gives us {n_features} features per image.")
print(
    f"There are {n_classes} unique targets (classes) in the dataset, of which {n_classes_larger_than_5} have at least five images, and {n_classes_larger_than_10} have at least ten images."
)
print(
    f"There are {n_images_from_classes_smaller_than_5} images from classes with less than five images, and {n_images_from_classes_smaller_than_10} images from classes with less than ten images."
)

# %%
original_images.shape

# %%
plot_images = original_images[target_names_first_index[0:10]]
label_array = np.array(target_names)[target_names_first_index[0:10]]

# Create a figure with a 2x5 grid of subplots
fig, axes = plt.subplots(2, 5, figsize=(12, 6))

# Iterate over the subplots and plot each image with its corresponding label
for i, ax in enumerate(axes.flat):
    ax.imshow(plot_images[i])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(str(label_array[i]))

# Show the figure
plt.show()

# %%
original_images.shape

# %%
# the channels: RGB
original_images[0, 0, 0, :]

# %%
plt.imshow(original_images[0])

# %%
plt.hist(original_images[0, :, :, 0])


# %%
def plot_original_faces(indices, max_faces=None, start=0):
    # Get the relevant images from the original_images array
    relevant_images = original_images[indices[start:]]

    # If max_faces is set, limit the number of images to plot
    if max_faces is not None:
        relevant_images = relevant_images[:max_faces]

    # Determine the number of rows and columns to use in the plot
    num_images = relevant_images.shape[0]
    num_cols = min(num_images, 5)
    num_rows = int(np.ceil(num_images / num_cols))

    # Create the plot and add subplots for each image
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    axs = axs.flatten()
    for i, img in enumerate(relevant_images):
        axs[i].imshow(img)
        axs[i].axis("off")

    # Remove any unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    plt.show()


# %% [markdown]
# ## Preprocess the images

# %% [markdown]
# ## Face detection
# Based on BlazeFace, taken inspiration from [this repo](https://github.com/ibaiGorordo/BlazeFace-TFLite-Inference).

# %%
from importlib import reload
from src.blazeface import BlazeFace, BlazeFaceResults

# reload(src.BlazeFace.BlazeFace)
# reload(BlazeFaceResults)

# %%
blazeface_model = BlazeFace(model_type="sparse")
blazeface_model.input_details

# %%
original_test_image = original_images[0]
test_image = blazeface_model.preprocess_input(original_test_image)

# %%
plt.imshow(test_image.numpy().squeeze()) if test_image.shape[0] == 1 else plt.imshow(
    test_image[0]
)

# %% [markdown]
# #### Extract detections

# %%
# import sys
# np.set_printoptions(threshold=sys.maxsize)

# %%
face_detection_results = blazeface_model.detect_faces(original_test_image)

# %%
input_tensor = blazeface_model.preprocess_input(original_test_image)

# Perform inference on the image
output0, output1 = blazeface_model.inference(input_tensor)

# Filter scores based on the detection scores
scores, good_detection_indices = blazeface_model.filter_detections(output1)

# %%
# np.set_printoptions(threshold=np.inf)
print(f"Raw scores:")
output1[output1 > -4]

# %%
output1[output1 > 0]
# Apparently 0.85 is the threshold for a good detection

# %%
print(f"Filtered (with sigmoid) scores:")
scores

# %%
output0.shape

# %%
print(
    f"50 examples of the first output tensor, which contains absolute the x offset of the bounding box for each face:"
)
output0[:50, 0]

# %%
print(
    f"50 examples of the second output tensor, which contains the absolute y offset of the bounding box for each face:"
)
output0[:50, 1]

# %%
print(
    f"50 examples of the third output tensor, which contains the absolute width of the bounding box for each face:"
)
output0[:50, 2]

# %%
print(
    f"50 examples of the fourth output tensor, which contains the absolute height of the bounding box for each face:"
)
output0[:50, 3]

# %%
print(f"fifth output tensor, which contains the x coordinate of first keypoint:")
output0[:50, 4]

# %%
print(f"sixth output tensor, which contains the y coordinate of first keypoint:")
output0[:50, 5]

# %%
idx = 7
print(f"Overview of anchor nummber {idx}:")
print(f"Anchor x_center: {blazeface_model.anchors[idx].x_center}")
print(f"Anchor y_center: {blazeface_model.anchors[idx].y_center}")
print(f"Anchor width: {blazeface_model.anchors[idx].w}")
print(f"Anchor height: {blazeface_model.anchors[idx].h}")


# %%
face_detection_results.boxes

# %%
face_detection_results.keypoints

# %%
face_detection_results.scores

# %%
plt.imshow(original_test_image)

# %%
plot_test_image = blazeface_model.draw_detections_single_image(
    original_test_image, face_detection_results
)

# %%
plt.imshow(plot_test_image)

# %% [markdown]
# #### Test detections

# %%
if not skip_intermediate_demo:
    blazeface_dense_model = BlazeFace(model_type="dense", score_threshold=0.6)

    face_detection_results = []
    detected_faces_count = 0
    detected_faces_indices = []
    undetected_faces_count = 0
    undetected_faces_indices = []
    double_detected_faces_count = 0
    double_detected_faces_indices = []

    for i in tqdm(range(original_images.shape[0])):
        face_detection_results.append(
            blazeface_dense_model.detect_faces(original_images[i])
        )
        if face_detection_results[-1].boxes.shape[0] > 0:
            detected_faces_count += 1
            detected_faces_indices.append(i)
            if face_detection_results[-1].boxes.shape[0] > 1:
                double_detected_faces_count += 1
                double_detected_faces_indices.append(i)

        else:
            undetected_faces_count += 1
            undetected_faces_indices.append(i)

    print(f"For the dense model:")
    print(
        f"Detected faces: {detected_faces_count} out of {original_images.shape[0]} ({detected_faces_count/original_images.shape[0]*100:.2f}%). Undetected faces: {undetected_faces_count} out of {original_images.shape[0]} ({undetected_faces_count/original_images.shape[0]*100:.2f}%)."
    )
    print(
        f"Double detected faces: {double_detected_faces_count} out of {original_images.shape[0]} ({double_detected_faces_count/original_images.shape[0]*100:.2f}%)."
    )

# %%
if not skip_intermediate_demo:
    blazeface_model = BlazeFace(model_type="sparse", score_threshold=0.6)

    face_detection_results = []
    detected_faces_count = 0
    detected_faces_indices = []
    undetected_faces_count = 0
    undetected_faces_indices = []
    double_detected_faces_count = 0
    double_detected_faces_indices = []

    for i in tqdm(range(original_images.shape[0])):
        face_detection_results.append(blazeface_model.detect_faces(original_images[i]))
        if face_detection_results[-1].boxes.shape[0] > 0:
            detected_faces_count += 1
            detected_faces_indices.append(i)
            if face_detection_results[-1].boxes.shape[0] > 1:
                double_detected_faces_count += 1
                double_detected_faces_indices.append(i)

        else:
            undetected_faces_count += 1
            undetected_faces_indices.append(i)

    print(
        f"Detected faces: {detected_faces_count} out of {original_images.shape[0]} ({detected_faces_count/original_images.shape[0]*100:.2f}%). Undetected faces: {undetected_faces_count} out of {original_images.shape[0]} ({undetected_faces_count/original_images.shape[0]*100:.2f}%)."
    )
    print(
        f"Double detected faces: {double_detected_faces_count} out of {original_images.shape[0]} ({double_detected_faces_count/original_images.shape[0]*100:.2f}%)."
    )


# %%
def plot_face_detections(indices, title, images=original_images, labels=target_names):
    plot_images = images[indices]
    label_array = np.array(labels)[indices]

    num_images = plot_images.shape[0]

    for i in range(num_images):
        detection_result = blazeface_model.detect_faces(plot_images[i])
        plot_images[i] = blazeface_model.draw_detections_single_image(
            plot_images[i], detection_result
        )

    # Calculate the number of rows and columns for the grid
    cols = min(5, num_images)
    rows = int(np.ceil(num_images / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

    # Customize the background color and text color
    bg_color = "#222222"
    text_color = "#ffffff"

    fig.patch.set_facecolor(bg_color)

    for i, ax in enumerate(axes.flat):
        if i < num_images:
            # Plot the image
            if plot_images.shape[-1] == 3:
                ax.imshow(plot_images[i].astype(np.uint8))
            else:
                ax.imshow(plot_images[i], cmap="gray")

            # Set the title with the corresponding label
            ax.set_title(f"{label_array[i]}", fontsize=12, color=text_color, y=1.02)

        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Set the background color for each subplot
        ax.set_facecolor(bg_color)

        # Set the color of the spines (border of the subplots)
        ax.spines["bottom"].set_color(text_color)
        ax.spines["top"].set_color(text_color)
        ax.spines["right"].set_color(text_color)
        ax.spines["left"].set_color(text_color)

    fig.suptitle(title, fontsize=20, y=1.02 + 0.02 * rows, color=text_color)
    plt.tight_layout(rect=[0, 0, 1, 1 - 0.04 * rows])
    plt.show()


# %%
if not skip_intermediate_demo:
    plot_face_detections(target_names_first_index, "Detected faces")

# %%
if not skip_intermediate_demo:
    plot_face_detections(undetected_faces_indices, "Undetected faces")

# %%
if not skip_intermediate_demo:
    plot_face_detections(double_detected_faces_indices, "Double face detections")

# %% [markdown]
# ## Face alignment
#
# For now I'm simply following [this github](https://sefiks.com/2020/02/23/face-alignment-for-face-recognition-in-python-within-opencv/) script from insightface.

# %%
from src.face_alignment import SimpleFaceAlignment

# %%
test_image = original_images[18]
test_image_face_detection_result = blazeface_model.detect_faces(test_image)
plt.imshow(test_image)

# %%
face_aligner = SimpleFaceAlignment()

# %%
test_aligned_image_results = face_aligner.crop_and_align(
    test_image, test_image_face_detection_result
)
test_aligned_image_results.aligned_faces[0].shape

# %%
plt.imshow(test_aligned_image_results.aligned_faces[0])

# %%
test_aligned_image_results

# %% [markdown]
# #### Test alignments

# %%
if not skip_intermediate_demo:
    face_detection_results = []
    face_alignment_results = []

    for idx in tqdm(range(original_images.shape[0])):
        image = original_images[idx]
        face_detection_results.append(blazeface_model.detect_faces(image))
        if (face_detection_results[-1].boxes.shape[0] > 0) & (
            face_detection_results[-1].boxes.shape[0] < 2
        ):
            face_alignment_results.append(
                face_aligner.crop_and_align(image, face_detection_results[idx])
            )
        else:
            face_alignment_results.append(None)


# %%
def plot_aligned_faces(
    aligned_face_results_list, max_faces=None, indices=None, start=0
):
    if indices is not None:
        aligned_face_results_list = [aligned_face_results_list[i] for i in indices]

    aligned_faces = [
        face
        for aligned_face_results in aligned_face_results_list
        if aligned_face_results is not None
        for face in aligned_face_results.aligned_faces
    ]

    if start > 0:
        aligned_faces = aligned_faces[start:]

    if max_faces is not None:
        aligned_faces = aligned_faces[:max_faces]

    num_faces = len(aligned_faces)
    cols = min(5, num_faces)
    rows = int(np.ceil(num_faces / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
    fig.suptitle("Aligned Faces", fontsize=20, y=1.02)

    for i, ax in enumerate(axes.flat):
        if i < num_faces:
            if aligned_faces[i].shape[-1] == 3:
                ax.imshow(aligned_faces[i].astype(np.uint8))
            else:
                ax.imshow(aligned_faces[i], cmap="gray")
            ax.set_title(f"Face {i + 1 + start}")

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.show()


# %%
if not skip_intermediate_demo:
    plot_aligned_faces(face_alignment_results, max_faces=50)

# %%
if not skip_intermediate_demo:
    plot_aligned_faces(
        face_alignment_results, max_faces=50, indices=target_names_first_index
    )

# %% [markdown]
# ## Face transformation
#
# Follows [this](https://github.com/deepinsight/insightface/blob/be3f7b3e0e635b56d903d845640b048247c41c90/common/face_align.py) example.

# %%
from src.face_alignment import ArcFaceAlignment

# %%
test_image = original_images[25]
plt.imshow(test_image)

# %%
test_image_face_detection_result = blazeface_model.detect_faces(test_image)
if not skip_intermediate_demo:
    test_image_face_detection_result

# %%
test_image.shape

# %%
test_image_landmarks_relative_full = test_image_face_detection_result.keypoints[0][:4]
test_image_landmarks_absolute_full = (test_image_landmarks_relative_full * 250).astype(
    np.int32
)
if not skip_intermediate_demo:
    test_image_landmarks_relative_full

# %%
test_image_face = test_image.copy()
test_image_face = blazeface_model.draw_detections_single_image(
    test_image_face, test_image_face_detection_result
)
plt.imshow(test_image_face)

# %%
crop_face = SimpleFaceAlignment()
test_image_cropped, xmin, ymin = crop_face.crop(
    test_image, test_image_face_detection_result, 250, 250
)
test_image_cropped = test_image_cropped[0][:112, :112]
plt.imshow(test_image_cropped)

# %%
test_image = original_images[30]
test_image_face_detection_result = blazeface_model.detect_faces(test_image)

# %%
arcface_align = ArcFaceAlignment()
test_image_align_result = arcface_align.crop_and_align(
    test_image, test_image_face_detection_result
)

print("Done with script")
