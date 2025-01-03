import cv2
import mediapipe as mp
from landmark_geometry import recognize_letter
import time
from text_to_speech import create_synthesizer
from autocorrection import create_autocorrector

def is_right_hand(landmarks, mirrored=True):
    if mirrored:
        return landmarks[mp_hands.HandLandmark.THUMB_TIP].x < landmarks[mp_hands.HandLandmark.WRIST].x
    else:
        return landmarks[mp_hands.HandLandmark.THUMB_TIP].x > landmarks[mp_hands.HandLandmark.WRIST].x


# Così controllo e inserisco lo spazio 
def is_hand_open(landmarks_dict):
    tips = [mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP, 
            mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.PINKY_TIP]
    mcps = [mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
            mp_hands.HandLandmark.RING_FINGER_MCP, mp_hands.HandLandmark.PINKY_MCP]
    
    fingers_up = all(landmarks_dict[tip].y < landmarks_dict[mcp].y - 0.1 for tip, mcp in zip(tips, mcps))
    fingers_spread = all(abs(landmarks_dict[tips[i]].x - landmarks_dict[tips[i+1]].x) > 0.04 for i in range(len(tips)-1))
    thumb_up = landmarks_dict[mp_hands.HandLandmark.THUMB_TIP].y < landmarks_dict[mp_hands.HandLandmark.THUMB_MCP].y
    
    return fingers_up and fingers_spread and thumb_up


# Inizializzazione MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

last_letter = ""
letter_start_time = 0
saved_letters = []
hand_detection_start_time = None
two_hands_start_time = None

space_start_time = None
waiting_for_space = False

LETTER_SAVE_DELAY = 1.5  
HAND_DETECTION_DELAY = 1.5
RESET_DELAY = 2
SPACE_DELAY = 1.2  

phrase_shown = False  
is_resetting = False  
current_phrase = ""  
detection_started = False  

corrected_phrase = ""  # Nuova variabile per la frase corretta
corrections = []      # Lista per le correzioni

candidates_dict = {}     
MAX_CANDIDATES_SHOWN = 5 

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

synthesizer = create_synthesizer()
autocorrector = create_autocorrector() 

while capture.isOpened():
    success, image = capture.read()
    if not success:
        print("Ignoring empty camera frame.")
        break
    
    # Conversione dell'immagine in RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = hands.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    current_time = time.time()
    
   
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 2:  # Controllo se ci sono due mani (per il reset)
        if not is_resetting: 
            two_hands_start_time = current_time
            is_resetting = True
        
        elif current_time - two_hands_start_time >= RESET_DELAY: #Reset completo
            
            saved_letters = []
            current_phrase = ""
            phrase_shown = False
            is_resetting = False
            corrected_phrase = ""  # Reset della frase corretta
            corrections = []       # Reset delle correzioni
            
            two_hands_start_time = None
            detection_started = False
            hand_detection_start_time = None
            

            space_start_time = None
            waiting_for_space = False
        
        else: #Parte il countdown per il reset
            time_left = RESET_DELAY - (current_time - two_hands_start_time)
            cv2.putText(image, f"Reset in: {time_left:.1f}s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        if current_phrase:  
            cv2.putText(image, f"Frase Composta: {current_phrase}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            if corrected_phrase and corrected_phrase != current_phrase:
                cv2.putText(image, f"Frase Corretta: {corrected_phrase}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
    
    elif results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:  #Se c'è una mano
        is_resetting = False
        two_hands_start_time = None

        if current_phrase and not is_resetting:
            cv2.putText(image, "Prima di iniziare il riconoscimento bisogna resettare la frase precedente", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        else:

            if not detection_started:
                if hand_detection_start_time is None:
                    hand_detection_start_time = current_time
                
                time_left = HAND_DETECTION_DELAY - (current_time - hand_detection_start_time)
                
                if time_left > 0:
                    cv2.putText(image, f"Inizio riconoscimento in: {time_left:.1f}s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) # Mostro il countdown per l'inizio del riconoscimento
                else:
                    detection_started = True
                    
        if detection_started and not phrase_shown:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS) #Disegno dei landmark della mano
            landmarks_dict = {l: hand_landmarks.landmark[l.value] for l in mp_hands.HandLandmark}
            
            
            
            # Controllo se la mano è aperta per inserire uno spazio
            if is_hand_open(landmarks_dict):
                
                if not waiting_for_space:
                    space_start_time = current_time
                    waiting_for_space = True
                
                elif current_time - space_start_time >= SPACE_DELAY:
                    
                    if not saved_letters or saved_letters[-1] != " ":  # Prevengo spazi multipli
                        saved_letters.append(" ")
                    
                    waiting_for_space = False
                    space_start_time = None
                
                else:
                    time_left = SPACE_DELAY - (current_time - space_start_time)
                    cv2.putText(image, f"Spazio in: {time_left:.1f}s", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            else:
                waiting_for_space = False
                space_start_time = None
                current_letter = recognize_letter(landmarks_dict)
                
                if current_letter != last_letter:  #Se la lettera è cambiata si resetta il timer
                    letter_start_time = current_time
                    last_letter = current_letter
                
                elif current_letter and (current_time - letter_start_time >= LETTER_SAVE_DELAY):
                        saved_letters.append(current_letter)
                        synthesizer.speak_letter(current_letter) 
                        letter_start_time = current_time
                
                if current_letter:
                    time_left = max(0, LETTER_SAVE_DELAY - (current_time - letter_start_time))
                    
                    cv2.putText(image, f"Lettera: {current_letter} ({time_left:.1f}s)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    cv2.putText(image, f"Lettere Salvate: {''.join(saved_letters)}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        
        if current_phrase:  # Visualizzazione frasi
            cv2.putText(image, f"Frase Composta: {current_phrase}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            if corrected_phrase and corrected_phrase != current_phrase:
                cv2.putText(image, f"Frase Corretta: {corrected_phrase}", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    else:  #Se non ci sono mani
        is_resetting = False
        two_hands_start_time = None
        
        if detection_started and saved_letters and not phrase_shown:  #Salvo e mostro la lettera composta
                current_phrase = "".join(saved_letters)
                
                # Applico l'autocorrezione
                corrected_phrase, corrections, candidates_dict = autocorrector.correct_phrase(current_phrase)
                phrase_shown = True
                detection_started = False
                hand_detection_start_time = None
                
                if corrections:  
                    synthesizer.speak_phrase(f"Ha composto: {current_phrase}")
                    time.sleep(0.3)  # Piccola pausa tra le frasi
                    synthesizer.speak_phrase(f"Forse intendeva: {corrected_phrase}")
                else:
                    synthesizer.speak_phrase(current_phrase)
        
        if not detection_started: #reset del timer per la detection delle mani nel caso in cui le faccio sparire dell'inquadratura
            hand_detection_start_time = None
        
        if current_phrase:  # Visualizzazione frasi
                y_offset = 170       
                cv2.putText(image, f"Frase Composta: {current_phrase}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                if corrected_phrase and corrected_phrase != current_phrase:
                    cv2.putText(image, f"Frase Corretta: {corrected_phrase}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # Mostro i candidati per ogni parola corretta
                    for word, candidates in candidates_dict.items():
                        if candidates:
                            shown_candidates = candidates[:MAX_CANDIDATES_SHOWN]
                            candidates_text = f"Candidati per '{word}': {', '.join(shown_candidates)}"
                            
                            if len(candidates) > MAX_CANDIDATES_SHOWN:
                                candidates_text += f" e altri {len(candidates) - MAX_CANDIDATES_SHOWN}"
                            
                            cv2.putText(image, candidates_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
                            
                            y_offset+=30
                            

    cv2.imshow('Riconoscimento LIS MediaPipe', image)
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

capture.release()
cv2.destroyAllWindows()
synthesizer.cleanup()
#autocorrector.cleanup()
