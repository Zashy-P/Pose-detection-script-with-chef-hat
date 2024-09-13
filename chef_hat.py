import cv2
import numpy as np
from pyvista import Plotter

# Function to overlay the 3D hat on the output image
def overlay_chef_hat(output_image, chef_hat, face_landmarks):
    # Extract the landmarks 
    forehead_landmark = face_landmarks.landmark[10] 
    chin_landmark = face_landmarks.landmark[152]  
    left_ear_landmark = face_landmarks.landmark[234] 
    right_ear_landmark = face_landmarks.landmark[454]  

    # convert coordinates into numpy arrays to perform vector operations
    forehead_point = np.array([forehead_landmark.x, forehead_landmark.y, forehead_landmark.z])
    chin_point = np.array([chin_landmark.x, chin_landmark.y, chin_landmark.z])
    left_ear_point = np.array([left_ear_landmark.x, left_ear_landmark.y, left_ear_landmark.z])
    right_ear_point = np.array([right_ear_landmark.x, right_ear_landmark.y, right_ear_landmark.z])

    # Calculate the face vector
    face_vector = forehead_point - chin_point

    # Normalize the face vector
    face_vector_normalized = face_vector / np.linalg.norm(face_vector)

    # Calculate the vector from left ear to right ear
    ear_vector = right_ear_point - left_ear_point

    # Position the hat above the forehead
    hat_position = forehead_point + 0.09 * face_vector_normalized

    # Calculate the orientation of the hat
    hat_orientation = np.eye(3)
    hat_orientation[:, 2] = face_vector_normalized  # Z-axis aligned with face vector
    hat_orientation[:, 0] = ear_vector / np.linalg.norm(ear_vector)  # X-axis aligned with ear vector
    hat_orientation[:, 1] = np.cross(hat_orientation[:, 2], hat_orientation[:, 0])  # Y-axis perpendicular to X and Z
    
    # Ensure the hat is upright by adjusting the orientation matrix
    up_vector = np.array([0, 1, 0])  
    hat_orientation[:, 1] = up_vector
    hat_orientation[:, 0] = np.cross(hat_orientation[:, 1], hat_orientation[:, 2])
    hat_orientation[:, 0] /= np.linalg.norm(hat_orientation[:, 0])
    hat_orientation[:, 1] = np.cross(hat_orientation[:, 2], hat_orientation[:, 0])

    # Create the transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = hat_orientation
    transformation_matrix[:3, 3] = hat_position

    # Apply the transformation to the 3D hat model
    transformed_hat = chef_hat.copy()
    transformed_hat.transform(transformation_matrix)

    # Scale the hat to make it smaller
    scale_factor = 0.03  
    transformed_hat.scale(scale_factor, inplace=True)

    # Render the transformed hat model onto the output image
    plotter = Plotter(off_screen=True)
    plotter.add_mesh(transformed_hat)
    plotter.camera_position = [(0, 0, -1), (0, 0, 0), (0, -1, 0)]  # Set the camera position to view the hat from the angle u want it or way
    plotter.show(auto_close=False)

    # Capture the rendered image from the plotter
    rendered_image = plotter.screenshot(transparent_background=True)
    plotter.close()

    # Convert the rendered image to the correct format
    rendered_image = cv2.cvtColor(rendered_image, cv2.COLOR_RGBA2BGRA)

     # Calculate the position to overlay the hat image
    x_offset = int(hat_position[0] * output_image.shape[1] - rendered_image.shape[1] / 2)
    y_offset = int(hat_position[1] * output_image.shape[0] - rendered_image.shape[0] / 2)

    # Ensure the transformed hat image fits within the output image boundaries
    y1, y2 = max(0, y_offset), min(output_image.shape[0], y_offset + rendered_image.shape[0])
    x1, x2 = max(0, x_offset), min(output_image.shape[1], x_offset + rendered_image.shape[1])

    # Ensure the hat image fits within the output image boundaries
    hat_y1, hat_y2 = max(0, -y_offset), min(rendered_image.shape[0], output_image.shape[0] - y_offset)
    hat_x1, hat_x2 = max(0, -x_offset), min(rendered_image.shape[1], output_image.shape[1] - x_offset)

    # Overlay the rendered hat image onto the output image
    alpha = rendered_image[hat_y1:hat_y2, hat_x1:hat_x2, 3] / 255.0
    for c in range(0, 3):
        output_image[y1:y2, x1:x2, c] = alpha * rendered_image[hat_y1:hat_y2, hat_x1:hat_x2, c] + (1 - alpha) * output_image[y1:y2, x1:x2, c]
