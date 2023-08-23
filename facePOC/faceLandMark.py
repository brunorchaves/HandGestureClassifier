#@markdown We implemented some functions to visualize the face landmark detection results. <br/> Run the following cell to activate the functions.

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import mouse
import numpy as np

def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

def plot_face_blendshapes_bar_graph(face_blendshapes):
  # Extract the face blendshapes category names and scores.
  face_blendshapes_names = [face_blendshapes_category.category_name for face_blendshapes_category in face_blendshapes]
  face_blendshapes_scores = [face_blendshapes_category.score for face_blendshapes_category in face_blendshapes]
  # The blendshapes are ordered in decreasing score value.
  face_blendshapes_ranks = range(len(face_blendshapes_names))

  fig, ax = plt.subplots(figsize=(12, 12))
  bar = ax.barh(face_blendshapes_ranks, face_blendshapes_scores, label=[str(x) for x in face_blendshapes_ranks])
  ax.set_yticks(face_blendshapes_ranks, face_blendshapes_names)
  ax.invert_yaxis()

  # Label each bar with values
  for score, patch in zip(face_blendshapes_scores, bar.patches):
    plt.text(patch.get_x() + patch.get_width(), patch.get_y(), f"{score:.4f}", va="top")

  ax.set_xlabel('Score')
  ax.set_title("Face Blendshapes")
  plt.tight_layout()
  plt.show()


# img = cv2.imread("image.jpg")
# cv2.imshow(img)

# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
cap = cv2.VideoCapture(0)

# STEP 2: Create an FaceLandmarker object.
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
     # Delete the previously saved JPEG image if it exists
    if os.path.exists('output.jpg'):
        os.remove('output.jpg')    
    # Convert the OpenCV frame to an mp.Image object.
    success, jpeg_data = cv2.imencode('.jpg', frame)
    if not success:
        print("Failed to encode frame as JPEG")
        break

    # Convert JPEG data to bytes
    jpeg_bytes = jpeg_data.tobytes()

    # Save the bytes to a JPEG file
    with open('output.jpg', 'wb') as file:
        file.write(jpeg_bytes)

    image = mp.Image.create_from_file('output.jpg')

    # STEP 4: Detect face landmarks from the input image.
    detection_result = detector.detect(image)

    # Display the frame with recognized gesture.
    # cv2.imshow('Face landmark', frame)
    
    #Drawing:
    # create a mask
    img = cv2.imread('output.jpg')
    mask = np.zeros(img.shape[:2], np.uint8)
    mask[100:250, 150:450] = 255
    # compute the bitwise AND using the mask

    # STEP 5: Process the detection result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    bgr_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    masked_img = cv2.bitwise_and(bgr_image,bgr_image,mask = mask)
    cv2.imshow("Annotade Image", masked_img)
    # plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0])
    # print(detection_result.face_blendshapes[0])
    blendshapes_data = detection_result.face_blendshapes[0]
    # Find and extract the "eyeBlinkLeft" category and its score
    eye_blink_left = None
    mouth_close  = None

    # Find and extract the "eyeBlinkLeft" and "mouthClose" categories and their scores
    for category in blendshapes_data:
        if category.category_name == 'eyeBlinkLeft':
            eye_blink_left = category
        elif category.category_name == 'mouthClose':
            mouth_close = category

    # Process and print the results
    if eye_blink_left is not None:
        eye_blink_left_name = eye_blink_left.category_name
        eye_blink_left_score = eye_blink_left.score
        print(f"{eye_blink_left_name}: {eye_blink_left_score}")
    else:
        print("eyeBlinkLeft not found in blendshapes data")

    if mouth_close is not None:
        mouth_close_name = mouth_close.category_name
        mouth_close_score = mouth_close.score
        print(f"{mouth_close_name}: {mouth_close_score}")
    else:
        print("mouthClose not found in blendshapes data")
    
    # else:
    #     print("eyeBlinkLeft not found in blendshapes data")
    # print(detection_result.facial_transformation_matrixes)

    # if(eye_blink_left_score >= 0.58):
    #     mouse.click('left')
    # if(mouth_close_score >= 0.0005):
        # mouse.click('right')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# # STEP 4: Detect face landmarks from the input image.
# detection_result = detector.detect(image)

# # STEP 5: Process the detection result. In this case, visualize it.
# annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
# bgr_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
# cv2.imshow("Annotade Image", bgr_image)
# plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0])
# # print(detection_result.face_blendshapes[0])
# print(detection_result.facial_transformation_matrixes)