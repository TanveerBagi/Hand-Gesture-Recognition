import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

#webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6) as hands:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      continue


    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:

          def gesture_classifier(hand_landmarks, image_width, image_height):
              thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
              thumb_ip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP]
              thumb_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP]

              index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
              middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
              ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
              pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

              index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
              middle_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]
              ring_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]
              pinky_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP]

              handedness = "Right" if thumb_tip.x < pinky_tip.x else "Left"

              thumb_extended = thumb_tip.y < thumb_ip.y < thumb_mcp.y  # Thumb pointing upwards

              index_extended = index_tip.y < index_mcp.y
              middle_extended = middle_tip.y < middle_mcp.y
              ring_extended = ring_tip.y < ring_mcp.y
              pinky_extended = pinky_tip.y < pinky_mcp.y


              if thumb_extended and not index_extended and not middle_extended and not ring_extended and not pinky_extended:
                  return "Thumbs Up"


              if thumb_extended and index_extended and middle_extended and ring_extended and pinky_extended:
                  return "Hello"


              if thumb_extended and not (index_extended or middle_extended or ring_extended or pinky_extended):
                  return "Yes"

              
              if not thumb_extended and index_extended and middle_extended and not ring_extended and not pinky_extended:
                  return "No"


              if  thumb_extended and index_extended and not middle_extended and not ring_extended and pinky_extended:
                  return "Peace"


              if thumb_extended and index_extended and middle_extended and not ring_extended and not pinky_extended:
                  return "Two"

              
              if thumb_extended and index_extended and not middle_extended and not ring_extended and not pinky_extended:
                  return "One"

              return "No other sign available"


          for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            gesture = gesture_classifier(hand_landmarks, image.shape[1], image.shape[0])
            print(f'Gesture: {gesture}')
            cv2.putText(image, f'Hand {idx + 1}: {gesture}', (10, 30 + (idx * 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
    
    cv2.imshow('Hand Gesture Recognition Project', image)

    if cv2.waitKey(5) & 0xFF == 27:
      break


cap.release()
cv2.destroyAllWindows()
