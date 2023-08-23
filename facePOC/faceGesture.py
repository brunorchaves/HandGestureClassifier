import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mouse
# # left click
# mouse.click('left')

# # right click
# mouse.click('right')

# # middle click
# mouse.click('middle')

model_path = 'rock_paper_scissors.task'

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)
# Open a video capture stream from the default camera (change the argument if using a different camera).
cap = cv2.VideoCapture(0)
# Load the input image.

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

    # Run gesture recognition.
    recognition_result = recognizer.recognize(image)

    if recognition_result.gestures:
        # Display the most likely gesture.
        top_gesture = recognition_result.gestures[0][0]
        print(f"Gesture recognized: {top_gesture.category_name} ({top_gesture.score})")
        if(top_gesture.category_name =="rock"):
            mouse.move(100, 100, absolute=True, duration=0.2)


    # Display the frame with recognized gesture.
    cv2.imshow('Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()