import cv2
import numpy as np
import os
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D,
    Embedding,
    Add,
)

# --- 1. Define Helper Functions & Variables ---

# MediaPipe setup
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
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

colors = [(245,117,16), (117,245,16), (16,117,245)]
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num % len(colors)], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    return output_frame

# --- 2. Build and Load the Model ---

# Model parameters from your notebook
embed_dim = 64
num_heads = 4
ff_dim = 64
sequence_length = 30
num_keypoints = 1662
actions = np.array(['hello', 'thanks', 'iloveyou'])

# --- THIS IS THE CORRECT PROCEDURAL ARCHITECTURE ---
# 1. Input Layer
inputs = Input(shape=(sequence_length, num_keypoints))

# 2. Project features
projected_inputs = Dense(embed_dim, name="feature_projection")(inputs)

# 3. Create and reshape positional embeddings
positions = tf.range(start=0, limit=sequence_length, delta=1)
position_embeddings = Embedding(input_dim=sequence_length, output_dim=embed_dim, name="position_embedding")(positions)
position_embeddings_broadcastable = tf.expand_dims(position_embeddings, axis=0)

# 4. Add positional embeddings to projected inputs
x = Add(name="add_positional_embedding")([projected_inputs, position_embeddings_broadcastable])

# 5. Transformer Block
attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, name="multi_head_attention")(x, x)
attention_output = Dropout(0.1)(attention_output)
x1 = Add(name="residual_connection_1")([x, attention_output])
x1 = LayerNormalization(epsilon=1e-6, name="layer_norm_1")(x1)

ffn_output = Dense(ff_dim, activation="relu", name="ffn_dense_1")(x1)
ffn_output = Dense(embed_dim, name="ffn_dense_2")(ffn_output)
ffn_output = Dropout(0.1)(ffn_output)
x2 = Add(name="residual_connection_2")([x1, ffn_output])
x2 = LayerNormalization(epsilon=1e-6, name="layer_norm_2")(x2)

# 6. Final Classification Head
x = GlobalAveragePooling1D()(x2)
x = Dense(20, activation="relu")(x)
x = Dropout(0.1)(x)
outputs = Dense(actions.shape[0], activation="softmax", name="output_layer")(x)

# 7. Create model
model = Model(inputs=inputs, outputs=outputs)
# --- END OF MODEL ARCHITECTURE ---


# Load the saved weights
model.load_weights('action_1000.h5')

print("Model loaded successfully.")

# --- 3. Real-Time Detection Loop ---

# ++-- CHANGE THIS VALUE IF YOUR CAMERA DOES NOT WORK --++
CAMERA_INDEX = 1
# ++-------------------------------------------------++

sequence = []
sentence = []
threshold = 0.8

cap = cv2.VideoCapture(CAMERA_INDEX)
# Set mediapipe model
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()
        
        # Check if frame is captured
        if not ret:
            print(f"Failed to grab frame from camera index {CAMERA_INDEX}. Exiting.")
            break

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        
        # Draw landmarks
        draw_styled_landmarks(image, results)
        
        # Prediction logic
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-sequence_length:]
        
        if len(sequence) == sequence_length:
            res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
            
            # Viz logic
            if res[np.argmax(res)] > threshold: 
                if len(sentence) > 0: 
                    if actions[np.argmax(res)] != sentence[-1]:
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            # Viz probabilities
            image = prob_viz(res, actions, image, colors)
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
