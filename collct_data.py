import cv2
import os
import numpy as np
import mediapipe as mp
import time

# ==== SETTINGS ====
DATA_PATH = os.path.join('MP_Data')  # Data storage path
actions = np.array(['hello', 'thanks', 'iloveyou'])  # Actions to collect
no_sequences = 30       # Number of sequences per action
sequence_length = 30    # Frames per sequence
cam_index = 0           # Camera index (change to 1 if external webcam)

# ==== MEDIAPIPE SETUP ====
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] 
                     for res in results.pose_landmarks.landmark]) \
                     if results.pose_landmarks else np.zeros((33, 4))
    face = np.array([[res.x, res.y, res.z] 
                     for res in results.face_landmarks.landmark]) \
                     if results.face_landmarks else np.zeros((468, 3))
    lh = np.array([[res.x, res.y, res.z] 
                   for res in results.left_hand_landmarks.landmark]) \
                   if results.left_hand_landmarks else np.zeros((21, 3))
    rh = np.array([[res.x, res.y, res.z] 
                   for res in results.right_hand_landmarks.landmark]) \
                   if results.right_hand_landmarks else np.zeros((21, 3))
    return np.concatenate([pose.flatten(), face.flatten(), lh.flatten(), rh.flatten()])

# ==== CREATE FOLDERS ====
for action in actions:
    for sequence in range(no_sequences):
        os.makedirs(os.path.join(DATA_PATH, action, str(sequence)), exist_ok=True)

# ==== START CAPTURE ====
cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)  # CAP_DSHOW avoids delay on Windows

if not cap.isOpened():
    print(f"Error: Could not open camera index {cam_index}")
    exit()

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    for action in actions:
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):

                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame. Check your camera.")
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                draw_styled_landmarks(image, results)

                # Show collection info
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)
                    cv2.putText(image, f'Collecting frames for {action} - Video {sequence}', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, f'Collecting frames for {action} - Video {sequence}', (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    cv2.imshow('OpenCV Feed', image)

                # Save keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), f"{frame_num}.npy")
                np.save(npy_path, keypoints)

                # Exit on 'q'
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    exit()

cap.release()
cv2.destroyAllWindows()
