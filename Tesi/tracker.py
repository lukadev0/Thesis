import cv2
import mediapipe as mp
from landmark_geometry import recognize_letter
import time

def is_right_hand(landmarks, mirrored=True):
    if mirrored:
        return landmarks[mp_hands.HandLandmark.THUMB_TIP].x < landmarks[mp_hands.HandLandmark.WRIST].x
    else:
        return landmarks[mp_hands.HandLandmark.THUMB_TIP].x > landmarks[mp_hands.HandLandmark.WRIST].x

# Inizializzazione MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils


last_letter = ""
letter_start_time = 0
saved_letters = []
hand_detection_start_time = None
two_hands_start_time = None

LETTER_SAVE_DELAY = 3  
HAND_DETECTION_DELAY = 1.5
RESET_DELAY = 2  

phrase_shown = False  
is_resetting = False  
current_phrase = ""  
detection_started = False  


capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)


while capture.isOpened():
    success, image = capture.read()
    if not success:
        print("Ignoring empty camera frame.")
        break
    
    #Conversione dell'immagine in RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    current_time = time.time()
    
    #Controllo se ci sono due mani (per il reset)
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:
        
        if not is_resetting:
            two_hands_start_time = current_time
            is_resetting = True
        
        elif current_time - two_hands_start_time >= RESET_DELAY: #Reset completo
            
            saved_letters = []
            current_phrase = ""
            phrase_shown = False
            is_resetting = False
            two_hands_start_time = None
            detection_started = False
            hand_detection_start_time = None
        
        else: #Parte il countdown per il reset
            
            time_left = RESET_DELAY - (current_time - two_hands_start_time)
            cv2.putText(image, f"Reset in: {time_left:.1f}s", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
       
        if current_phrase:  #Continua a mostrare frase corrente
            cv2.putText(image, f"Frase Corrente: {current_phrase}", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
   
    elif results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:  #Se c'è una mano
        is_resetting = False
        two_hands_start_time = None

        if not detection_started:
            if hand_detection_start_time is None:
                hand_detection_start_time = current_time
            
            time_left = HAND_DETECTION_DELAY - (current_time - hand_detection_start_time)
            if time_left > 0:
                cv2.putText(image, f"Inizio riconoscimento in: {time_left:.1f}s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) # Mostro il countdown per l'inizio del riconoscimento
            else:
                detection_started = True
                
        if detection_started and not phrase_shown:
            
            for hand_landmarks in results.multi_hand_landmarks:
                
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS) #Disegno dei landmark della mano
                landmarks_dict = {l: hand_landmarks.landmark[l.value] for l in mp_hands.HandLandmark}
                current_letter = recognize_letter(landmarks_dict)
                
                if current_letter != last_letter:  #Se la lettera è cambiata si resetta il timer
                    letter_start_time = current_time
                    last_letter = current_letter

                elif current_letter and (current_time - letter_start_time >= LETTER_SAVE_DELAY):
                    if not saved_letters or saved_letters[-1] != current_letter:
                        saved_letters.append(current_letter)
                        letter_start_time = current_time
                
                if current_letter:
                    time_left = max(0, LETTER_SAVE_DELAY - (current_time - letter_start_time)) 
                    
                    cv2.putText(image, f"Lettera: {current_letter} ({time_left:.1f}s)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    cv2.putText(image, f"Lettere Salvate: {''.join(saved_letters)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        
        if current_phrase:
            cv2.putText(image, f"Frase/Parola Composta: {current_phrase}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) # mostro la frase composta
    
    
    else:  #Se non ci sono mani
        is_resetting = False
        two_hands_start_time = None
        
        if detection_started and saved_letters and not phrase_shown:  #Salvo e mostro la lettera composta
            
            current_phrase = "".join(saved_letters)
            phrase_shown = True
            detection_started = False
            hand_detection_start_time = None
        
        if not detection_started: #reset del timer per la detection delle mani nel caso in cui le faccio sparire dell'inquadratura
            hand_detection_start_time = None
        
        if current_phrase:
            cv2.putText(image, f"Frase Composta: {current_phrase}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) #Faccio rimanere fino al reset la frase composta a schermo

    cv2.imshow('Riconoscimento LIS MediaPipe', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

capture.release()
cv2.destroyAllWindows()