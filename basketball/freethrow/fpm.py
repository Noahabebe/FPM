import cv2
import mediapipe as mp
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle  # To load pre-trained fatigue model

# Initialize MediaPipe pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Load the pre-trained fatigue model (RandomForest in this example)
with open('SPL-Open-Data/basketball/freethrow/fatigue_model.pkl', 'rb') as f:
    fatigue_model = pickle.load(f)

# Function to convert an RGB color to its HSV range with tolerance
def rgb_to_hsv_range(rgb_color, tolerance=40):
    rgb_color = np.uint8([[rgb_color]])  # Convert the RGB value to the proper format
    hsv_color = cv2.cvtColor(rgb_color, cv2.COLOR_RGB2HSV)[0][0]  # Convert to HSV
    
    lower_bound = np.array([hsv_color[0] - tolerance, 100, 100])  # Adjust tolerance for hue
    upper_bound = np.array([hsv_color[0] + tolerance, 255, 255])  # Saturation and value at max
    
    return lower_bound, upper_bound

# Function to detect team jersey color
def detect_team_jersey_color(frame, lower_hsv, upper_hsv):
    """Detect if the player is wearing the team jersey by color."""
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_hsv, upper_hsv)
    team_color_present = np.any(mask)  # Boolean for team color detection
    return team_color_present

# Function to detect ball
def detect_ball(frame):
    """Basic detection of the ball using contour detection (can be enhanced)."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    _, thresh = cv2.threshold(blurred_frame, 60, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Adjust area threshold for ball size
            return True  # Ball detected
    
    return False

# Capture video from default camera
cap = cv2.VideoCapture(0)

# Example: Define RGB color for team jersey (e.g., red team)
team_rgb_color = [255, 0, 0]  # Replace with any RGB value
lower_hsv, upper_hsv = rgb_to_hsv_range(team_rgb_color)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Convert the frame to RGB for MediaPipe processing
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Pose detection with MediaPipe
    result = pose.process(img_rgb)
    
    # Detect team jersey color
    is_team_member = detect_team_jersey_color(frame, lower_hsv, upper_hsv)
    
    # Detect if player is holding the ball
    ball_detected = detect_ball(frame)
    
    if result.pose_landmarks:
        # Draw landmarks
        mp.solutions.drawing_utils.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Extract keypoints for jump height calculation
        ankle_right = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]
        knee_right = result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]
        
        # Calculate vertical jump height in meters (approximation)
        jump_height = abs(ankle_right.z - knee_right.z)  # Z-coordinates give depth (approximating vertical height)
        
        # Display jump height on the frame
        cv2.putText(frame, f"Jump Height: {jump_height:.2f}m", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Predict fatigue based on features: [jump_height, ball_possession, is_team_member]
        fatigue_input = np.array([[jump_height, ball_detected, is_team_member]])
        predicted_fatigue = fatigue_model.predict(fatigue_input)[0]
        
        # Display the predicted fatigue level
        cv2.putText(frame, f"Predicted Fatigue: {predicted_fatigue:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Show the processed video frame
    cv2.imshow("Fatigue Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
