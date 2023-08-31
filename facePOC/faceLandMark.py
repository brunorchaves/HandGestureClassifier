#@markdown We implemented some functions to visualize the face landmark detection results. <br/> Run the following cell to activate the functions.

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import mouse
import numpy as np

mask_x_coord = 200
mask_y_coord = 100

class mouseCoord:
    def __init__(self, x,y):
        self.x = x
        self.y = y
    # getter method
    def get_x(self):
        return self.x  
    # setter method
    def set_x(self, value_x):
        self.x = value_x
    # getter method
    def get_y(self):
        return self.y
    # setter method
    def set_y(self, value_y):
        self.y = value_y

mousePos = mouseCoord(100,10)
# Define the click_event function for mouse clicks
def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        # print("Mouse clicked at coordinates (x={}, y={})".format(x, y))
        mousePos.set_x(x)
        mousePos.set_y(y)

# Initialize OpenCV window for mouse clicks
cv2.namedWindow("Annotated Image")
cv2.setMouseCallback("Annotated Image", click_event)
  
  
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
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
        # print("Failed to encode frame as JPEG")
        break

    # Convert JPEG data to bytes
    jpeg_bytes = jpeg_data.tobytes()

    # Save the bytes to a JPEG file
    with open('output.jpg', 'wb') as file:
        file.write(jpeg_bytes)

    image = mp.Image.create_from_file('output.jpg')

    # STEP 4: Detect face landmarks from the input image.
    detection_result = detector.detect(image)
    import re
    # print((detection_result.face_landmarks))
    # landMarksList=  str(detection_result.face_landmarks)
    # Extract each NormalizedLandmark instance using regular expression
    input_string = str(detection_result.face_landmarks)
    landmark_strings = re.findall(r'NormalizedLandmark\(.*?\)', input_string)

    # Define a class for NormalizedLandmark
    class NormalizedLandmark:
        def __init__(self, x, y, z, visibility, presence):
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)
            self.visibility = float(visibility)
            self.presence = float(presence)

    # Convert the extracted strings to NormalizedLandmark objects
    landmarks = []
    for landmark_str in landmark_strings:
        match = re.match(r'NormalizedLandmark\(x=(.*?), y=(.*?), z=(.*?), visibility=(.*?), presence=(.*?)\)', landmark_str)
        if match:
            x, y, z, visibility, presence = match.groups()
            landmark = NormalizedLandmark(x, y, z, visibility, presence)
            landmarks.append(landmark)

    # Now 'landmarks' is a list of NormalizedLandmark objects
    # print(landmarks[29].x)
    # print(landmarks[29].y)
    nose_landMarkCoord_denormalized_x = frame_width*(landmarks[29].x)
    nose_landMarkCoord_denormalized_y = frame_width*(landmarks[29].y)
    print(nose_landMarkCoord_denormalized_x)
    print(nose_landMarkCoord_denormalized_y)

    # Display the frame with recognized gesture.
    # cv2.imshow('Face landmark', frame)
    
    #Drawing:
    # create a mask
    img = cv2.imread('output.jpg')
    mask = np.zeros(img.shape[:2], np.uint8)
    # mask_x_coord= mousePos.get_x()
    # mask_y_coord= mousePos.get_y()
    mask_x_coord= int(nose_landMarkCoord_denormalized_x)
    mask_y_coord= int(nose_landMarkCoord_denormalized_y)
    # Define the size of the mask
    mask_x_size = 200
    mask_y_size = 300

    mask_x_half_size = int(mask_x_size/2)
    mask_y_half_size = int(mask_y_size/2)

    # Define the size of the mask

    # Apply the mask at the specified coordinates
    mask[mask_y_coord-mask_y_half_size:mask_y_coord+mask_y_half_size, mask_x_coord-mask_x_half_size:mask_x_coord+mask_x_half_size+50] = 255

    # compute the bitwise AND using the mask

    # STEP 5: Process the detection result. In this case, visualize it.
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    # print(detection_result)
    bgr_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    masked_img = cv2.bitwise_and(bgr_image,bgr_image,mask = mask)
    cv2.imshow("Annotated Image", masked_img)
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
        # print(f"{eye_blink_left_name}: {eye_blink_left_score}")
    # else:
        # print("eyeBlinkLeft not found in blendshapes data")

    if mouth_close is not None:
        mouth_close_name = mouth_close.category_name
        mouth_close_score = mouth_close.score
        # print(f"{mouth_close_name}: {mouth_close_score}")
    # else:
        # print("mouthClose not found in blendshapes data")
    


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