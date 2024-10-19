import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands

# Definizione dei landmark delle dita
WRIST = mp_hands.HandLandmark.WRIST
THUMB_TIP = mp_hands.HandLandmark.THUMB_TIP
THUMB_MCP = mp_hands.HandLandmark.THUMB_MCP
INDEX_FINGER_MCP = mp_hands.HandLandmark.INDEX_FINGER_MCP
INDEX_FINGER_PIP = mp_hands.HandLandmark.INDEX_FINGER_PIP
INDEX_FINGER_TIP = mp_hands.HandLandmark.INDEX_FINGER_TIP
MIDDLE_FINGER_MCP = mp_hands.HandLandmark.MIDDLE_FINGER_MCP
MIDDLE_FINGER_PIP = mp_hands.HandLandmark.MIDDLE_FINGER_PIP
MIDDLE_FINGER_TIP = mp_hands.HandLandmark.MIDDLE_FINGER_TIP
RING_FINGER_MCP = mp_hands.HandLandmark.RING_FINGER_MCP
RING_FINGER_PIP = mp_hands.HandLandmark.RING_FINGER_PIP
RING_FINGER_TIP = mp_hands.HandLandmark.RING_FINGER_TIP
PINKY_MCP = mp_hands.HandLandmark.PINKY_MCP
PINKY_PIP = mp_hands.HandLandmark.PINKY_PIP
PINKY_TIP = mp_hands.HandLandmark.PINKY_TIP

def get_distance(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def get_angle(p1, p2, p3):
    v1 = np.array([p1.x - p2.x, p1.y - p2.y])
    v2 = np.array([p3.x - p2.x, p3.y - p2.y])
    return np.abs(np.math.atan2(np.linalg.det([v1,v2]),np.dot(v1,v2)))

def finger_is_closed(landmarks, finger_tip, finger_mcp):
    return get_distance(landmarks[finger_tip], landmarks[WRIST]) < get_distance(landmarks[finger_mcp], landmarks[WRIST])

def finger_is_straight(landmarks, finger_mcp, finger_pip, finger_tip):
    angle = get_angle(landmarks[finger_mcp], landmarks[finger_pip], landmarks[finger_tip])
    return angle > 2.8  # Circa 160 gradi

def finger_is_curled(landmarks, finger_mcp, finger_pip, finger_tip):
    angle = get_angle(landmarks[finger_mcp], landmarks[finger_pip], landmarks[finger_tip])
    return angle < 1.5  # Circa 90 gradi

def fingers_are_close(landmarks, fingers):
    distances = [get_distance(landmarks[fingers[i]], landmarks[fingers[i+1]]) for i in range(len(fingers)-1)]
    return all(d < 0.1 for d in distances)

def is_A(landmarks):
    fingers_closed = all(finger_is_closed(landmarks, tip, mcp) for tip, mcp in [
        (INDEX_FINGER_TIP, INDEX_FINGER_MCP),
        (MIDDLE_FINGER_TIP, MIDDLE_FINGER_MCP),
        (RING_FINGER_TIP, RING_FINGER_MCP),
        (PINKY_TIP, PINKY_MCP)
    ])
    thumb_folded = get_distance(landmarks[THUMB_TIP], landmarks[INDEX_FINGER_MCP]) < 0.1
    return fingers_closed and thumb_folded

def is_B(landmarks):
    thumb_tip = landmarks[THUMB_TIP]
    index_tip = landmarks[INDEX_FINGER_TIP]
    middle_tip = landmarks[MIDDLE_FINGER_TIP]
    ring_tip = landmarks[RING_FINGER_TIP]
    pinky_tip = landmarks[PINKY_TIP]
    wrist = landmarks[WRIST]

    fingers_straight = all(finger_is_straight(landmarks, mcp, pip, tip) for mcp, pip, tip in [
        (INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_TIP),
        (MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP, MIDDLE_FINGER_TIP),
        (RING_FINGER_MCP, RING_FINGER_PIP, RING_FINGER_TIP),
        (PINKY_MCP, PINKY_PIP, PINKY_TIP)
    ])
    fingers_close = (get_distance(index_tip, pinky_tip) < 0.1)
    thumb_lower = (thumb_tip.y > index_tip.y)
    fingers_up = all(tip.y < wrist.y for tip in [index_tip, middle_tip, ring_tip, pinky_tip])
    thumb_not_folded = get_distance(thumb_tip, wrist) > get_distance(landmarks[THUMB_MCP], wrist)

    return fingers_straight and fingers_close and thumb_lower and fingers_up and thumb_not_folded

def is_C(landmarks):
    thumb_tip = landmarks[THUMB_TIP]
    index_tip = landmarks[INDEX_FINGER_TIP]
    middle_tip = landmarks[MIDDLE_FINGER_TIP]
    ring_tip = landmarks[RING_FINGER_TIP]
    pinky_tip = landmarks[PINKY_TIP]
    
    # distanza tra il pollice e l'indice
    thumb_index_distance = get_distance(thumb_tip, index_tip)
    
    # curvatura delle dita 
    fingers_curved = all(0.2 < get_angle(landmarks[mcp], landmarks[pip], landmarks[tip]) < 2.3 
                         for mcp, pip, tip in [
                             (INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_TIP),
                             (MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP, MIDDLE_FINGER_TIP),
                             (RING_FINGER_MCP, RING_FINGER_PIP, RING_FINGER_TIP),
                             (PINKY_MCP, PINKY_PIP, PINKY_TIP)
                         ])
    
    # Verifico della posizine del pollice rispetto alle altre dita
    thumb_position = landmarks[THUMB_TIP].x < landmarks[INDEX_FINGER_MCP].x
    
    # Verifico che indice, medio, anulare e mignolo siano allineati
    finger_tips_y = [index_tip.y, middle_tip.y, ring_tip.y, pinky_tip.y]
    max_y_diff = max(finger_tips_y) - min(finger_tips_y)
    fingers_aligned = max_y_diff < 0.05  # Soglia per l'allineamento verticale, posso cambiarlo a mio piacimento
    
    # Verifica che le dita siano raggruppate orizzontalmente
    finger_tips_x = [index_tip.x, middle_tip.x, ring_tip.x, pinky_tip.x]
    max_x_diff = max(finger_tips_x) - min(finger_tips_x)
    fingers_grouped = max_x_diff < 0.1  # Soglia per il raggruppamento orizzontale
    
    # Condizioni per riconoscere la lettera 
    thumb_index_condition = 0.08 < thumb_index_distance < 0.25
    
    return (thumb_index_condition and 
            fingers_curved and 
            thumb_position and 
            fingers_aligned and 
            fingers_grouped)

def is_D(landmarks):
    index_straight = finger_is_straight(landmarks, INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_TIP)
    other_fingers_touch_thumb = all(get_distance(landmarks[THUMB_TIP], landmarks[tip]) < 0.1 for tip in [
        MIDDLE_FINGER_TIP, RING_FINGER_TIP, PINKY_TIP
    ])
    return index_straight and other_fingers_touch_thumb 

def is_E(landmarks):
    all_fingers_curled = all(finger_is_curled(landmarks, mcp, pip, tip) for mcp, pip, tip in [
        (INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_TIP),
        (MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP, MIDDLE_FINGER_TIP),
        (RING_FINGER_MCP, RING_FINGER_PIP, RING_FINGER_TIP),
        (PINKY_MCP, PINKY_PIP, PINKY_TIP)
    ])
    thumb_hidden = landmarks[THUMB_TIP].y > landmarks[INDEX_FINGER_PIP].y
    return all_fingers_curled and thumb_hidden

def is_F(landmarks):
    thumb_tip = landmarks[THUMB_TIP]
    index_tip = landmarks[INDEX_FINGER_TIP]
    middle_tip = landmarks[MIDDLE_FINGER_TIP]
    
    
    thumb_index_touch = get_distance(thumb_tip, index_tip) < 0.05
    
    
    other_fingers_straight = all(finger_is_straight(landmarks, mcp, pip, tip) 
                                 for mcp, pip, tip in [
                                     (MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP, MIDDLE_FINGER_TIP),
                                     (RING_FINGER_MCP, RING_FINGER_PIP, RING_FINGER_TIP),
                                     (PINKY_MCP, PINKY_PIP, PINKY_TIP)
                                 ])
    
    
    index_middle_distance = get_distance(index_tip, middle_tip) > 0.1
    
    return thumb_index_touch and other_fingers_straight and index_middle_distance

def recognize_letter(landmarks):
    if is_A(landmarks):
        return 'A'
    elif is_B(landmarks):
        return 'B'
    elif is_C(landmarks):
        return 'C'
    elif is_D(landmarks):
        return 'D'
    elif is_E(landmarks):
        return 'E'
    elif is_F(landmarks):
        return 'F'
    else:
        return 'Unknown'