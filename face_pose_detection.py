import cv2
import mediapipe as mp
import numpy as np
import pyvista as pv
from chef_hat import overlay_chef_hat

# Initialize MediaPipe solutions for pose and face mesh
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define drawing specifications for landmarks (white color, small thickness, and small circle radius)
drawing_spec = mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)

# Start capturing video from the default camera
cap = cv2.VideoCapture(0)
# Specify the initial window size (for example, 1280x720)
initial_width = 1280
initial_height = 720

# Load the 3D chef hat model
chef_hat = pv.read("Chef Hat.ply")

# Create a named window that can be resized
cv2.namedWindow('MediaPipe Pose with Face Mesh & Chef Hat', cv2.WINDOW_NORMAL)

# Resize the window initially to the width and height specified
cv2.resizeWindow('MediaPipe Pose with Face Mesh & Chef Hat', initial_width, initial_height)

# Flag to toggle the appearance of the hat
show_hat = False

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB before processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False

    # Process the image and find the pose and face mesh
    pose_results = pose.process(image_rgb)
    face_results = face_mesh.process(image_rgb)

    # Draw the pose and face mesh on a black background
    # Create a black image with the same dimensions as the camera feed
    output_image = np.zeros(image.shape, dtype=np.uint8)

    # Draw the pose landmarks on the black image
    if pose_results.pose_landmarks:
        mp_drawing.draw_landmarks(
            output_image,
            pose_results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

      # Draw the face mesh landmarks on the black image and add the chef hat
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                output_image,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
        if show_hat:
            overlay_chef_hat(output_image, chef_hat, face_landmarks)

    # Display the output image with pose and face mesh
    cv2.imshow('MediaPipe Pose with Face Mesh & Chef Hat', output_image)

    key = cv2.waitKey(5) & 0xFF
    # Break the loop when 'ESC' key is pressed
    if key == 27:
        break
    # Toggle the hat by pressing the 'HOME' key
    elif key == 0: 
        show_hat = not show_hat

# Release the camera and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()
