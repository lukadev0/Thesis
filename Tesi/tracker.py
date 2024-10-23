import cv2
import mediapipe as mp
from landmark_geometry import recognize_letter

def is_right_hand(landmarks, mirrored=True):
    
    if mirrored:
        return landmarks[mp_hands.HandLandmark.THUMB_TIP].x < landmarks[mp_hands.HandLandmark.WRIST].x
    else:
        return landmarks[mp_hands.HandLandmark.THUMB_TIP].x > landmarks[mp_hands.HandLandmark.WRIST].x

# Inizializzazione MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils


capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

'''for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Webcam found at index {i}")
        cap.release()'''

while capture.isOpened():
    success, image = capture.read()
    if not success:
        print("Ignoring empty camera frame.")
        break
    
    # Conversione dell'immagine in RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)  # Immagine specchiata
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  

    # Inizializza una lista per memorizzare il testo da visualizzare
    display_texts = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Disegno dei landmark della mano
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Conversione dei landmark in un dizionario
            landmarks_dict = {l: hand_landmarks.landmark[l.value] for l in mp_hands.HandLandmark}
            
            recognized_letter = recognize_letter(landmarks_dict)
            #hand_side = "Destra" if is_right_hand(landmarks_dict) else "Sinistra"

            display_texts.append(f"Segno Visualizzato: {recognized_letter}")
    

    if len(display_texts) == 1:

        cv2.putText(image, display_texts[0], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    elif len(display_texts) == 2:
        cv2.putText(image, display_texts[0], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, display_texts[1], (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


    cv2.imshow('Riconoscimento LIS MediaPipe', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

capture.release()
cv2.destroyAllWindows()