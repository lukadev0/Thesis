import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands

# Definizione dei landmark delle dita
WRIST = mp_hands.HandLandmark.WRIST
THUMB_TIP = mp_hands.HandLandmark.THUMB_TIP
THUMB_MCP = mp_hands.HandLandmark.THUMB_MCP
THUMB_IP = mp_hands.HandLandmark.THUMB_IP
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
    
    return (fingers_closed and thumb_folded)

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
    fingers_close = (get_distance(index_tip, pinky_tip) < 0.15)
    thumb_lower = (thumb_tip.y > index_tip.y)
    fingers_up = all(tip.y < wrist.y for tip in [index_tip, middle_tip, ring_tip, pinky_tip])
    thumb_not_folded = get_distance(thumb_tip, wrist) > get_distance(landmarks[THUMB_MCP], wrist)
    
    thumb_bent = get_distance(thumb_tip, landmarks[INDEX_FINGER_MCP]) < 0.1

    return (fingers_straight and fingers_close and thumb_lower and 
            fingers_up and thumb_not_folded and thumb_bent)

def is_C(landmarks):
    thumb_tip = landmarks[THUMB_TIP]
    index_tip = landmarks[INDEX_FINGER_TIP]
    middle_tip = landmarks[MIDDLE_FINGER_TIP]
    ring_tip = landmarks[RING_FINGER_TIP]
    pinky_tip = landmarks[PINKY_TIP]
    
    
    thumb_index_distance = get_distance(thumb_tip, index_tip)
     
    fingers_curved = all(0.8 < get_angle(landmarks[mcp], landmarks[pip], landmarks[tip]) < 2.5
                         for mcp, pip, tip in [
                             (INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_TIP),
                             (MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP, MIDDLE_FINGER_TIP),
                             (RING_FINGER_MCP, RING_FINGER_PIP, RING_FINGER_TIP),
                             (PINKY_MCP, PINKY_PIP, PINKY_TIP)
                         ])
    
   
    thumb_position = landmarks[THUMB_TIP].x < landmarks[INDEX_FINGER_MCP].x
    
    
    finger_tips_y = [index_tip.y, middle_tip.y, ring_tip.y, pinky_tip.y]
    max_y_diff = max(finger_tips_y) - min(finger_tips_y)
    fingers_aligned = max_y_diff < 0.05  
    
    
    finger_tips_x = [index_tip.x, middle_tip.x, ring_tip.x, pinky_tip.x]
    max_x_diff = max(finger_tips_x) - min(finger_tips_x)
    fingers_grouped = max_x_diff < 0.1  
    
    
    thumb_index_condition = 0.08 < thumb_index_distance < 0.25
    
    return (thumb_index_condition and fingers_curved and thumb_position and fingers_aligned and fingers_grouped)

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
    
    thumb_folded = get_distance(landmarks[THUMB_TIP], landmarks[INDEX_FINGER_MCP]) < 0.12
    return ( all_fingers_curled and thumb_folded )

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
    
    return (thumb_index_touch and other_fingers_straight and index_middle_distance)

def is_G(landmarks):
    index_tip = landmarks[INDEX_FINGER_TIP]
    
    index_straight = finger_is_straight(landmarks, INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_TIP) 
    
    
    thumb_on_middle = get_distance(landmarks[THUMB_TIP], landmarks[MIDDLE_FINGER_TIP]) < 0.1
    
    middle_curled = finger_is_curled(landmarks, MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP, MIDDLE_FINGER_TIP)
    ring_curled = finger_is_curled(landmarks, RING_FINGER_MCP, RING_FINGER_PIP, RING_FINGER_TIP)
    pinky_curled = finger_is_curled(landmarks, PINKY_MCP, PINKY_PIP, PINKY_TIP)
    
    index_horizontal = (abs(index_tip.y - landmarks[INDEX_FINGER_MCP].y) < 0.1 and 
                       abs(index_tip.x - landmarks[INDEX_FINGER_MCP].x) > 0.12)

    return (index_straight and thumb_on_middle and middle_curled and ring_curled and pinky_curled and index_horizontal)

def is_H(landmarks):
    
    index_tip = landmarks[INDEX_FINGER_TIP]
    middle_tip = landmarks[MIDDLE_FINGER_TIP]
    
    index_straight = finger_is_straight(landmarks, INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_TIP)
    middle_straight = finger_is_straight(landmarks, MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP, MIDDLE_FINGER_TIP)
    
    fingers_straight = (get_distance(index_tip, middle_tip) < 0.1)
    
    other_fingers_curled = all(finger_is_curled(landmarks, mcp, pip, tip) for mcp, pip, tip in [
        (RING_FINGER_MCP, RING_FINGER_PIP, RING_FINGER_TIP),
        (PINKY_MCP, PINKY_PIP, PINKY_TIP)
    ])
    
    index_horizontal = (abs(index_tip.y - landmarks[INDEX_FINGER_MCP].y) < 0.1 and 
                       abs(index_tip.x - landmarks[INDEX_FINGER_MCP].x) > 0.15)
    
    middle_horizontal = (abs(middle_tip.y - landmarks[MIDDLE_FINGER_MCP].y) < 0.1 and
                        abs(middle_tip.x - landmarks[MIDDLE_FINGER_MCP].x) > 0.15)
    
    return (index_straight and middle_straight and 
            other_fingers_curled and fingers_straight and
            index_horizontal and middle_horizontal)

def is_I(landmarks):
    
    pinky_straight = finger_is_straight(landmarks, PINKY_MCP, PINKY_PIP, PINKY_TIP)
    pinky_pointing_up = landmarks[PINKY_TIP].y < landmarks[PINKY_MCP].y
    

    thumb_touches_index = get_distance(landmarks[THUMB_TIP], landmarks[INDEX_FINGER_PIP]) < 0.08
    
   
    other_fingers_curled = all(
        finger_is_curled(landmarks, mcp, pip, tip) for mcp, pip, tip in [
            (INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_TIP),
            (MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP, MIDDLE_FINGER_TIP),
            (RING_FINGER_MCP, RING_FINGER_PIP, RING_FINGER_TIP)
        ]
    )
    
   
    fingers_closed = all(
        landmarks[tip].y > landmarks[mcp].y for tip, mcp in [
            (INDEX_FINGER_TIP, INDEX_FINGER_MCP),
            (MIDDLE_FINGER_TIP, MIDDLE_FINGER_MCP),
            (RING_FINGER_TIP, RING_FINGER_MCP)
        ]
    )
    
    
    pinky_higher = all(
        landmarks[PINKY_TIP].y < landmarks[tip].y - 0.1 for tip in [
            INDEX_FINGER_TIP,
            MIDDLE_FINGER_TIP,
            RING_FINGER_TIP
        ]
    )

    return (pinky_straight and pinky_pointing_up and thumb_touches_index and other_fingers_curled and fingers_closed and pinky_higher)

def is_J(landmarks):
   
    pinky_straight = finger_is_straight(landmarks, PINKY_MCP, PINKY_PIP, PINKY_TIP)
    
   
    pinky_horizontal_and_low = (landmarks[PINKY_TIP].y > landmarks[THUMB_TIP].y and
                                abs(landmarks[PINKY_TIP].x - landmarks[PINKY_MCP].x) > 0.1)
    
   
    thumb_on_top = (landmarks[THUMB_TIP].y < landmarks[INDEX_FINGER_TIP].y and
                    abs(landmarks[THUMB_TIP].x - landmarks[INDEX_FINGER_MCP].x) < 0.1)

    middle_curled = finger_is_curled(landmarks, MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP, MIDDLE_FINGER_TIP)
    ring_curled = finger_is_curled(landmarks, RING_FINGER_MCP, RING_FINGER_PIP, RING_FINGER_TIP)
    index_curled = finger_is_curled(landmarks, INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_TIP)

    return (pinky_straight and pinky_horizontal_and_low and thumb_on_top and 
            middle_curled and ring_curled and index_curled)

def is_K(landmarks):
   
    thumb_straight = finger_is_straight(landmarks, THUMB_MCP, THUMB_IP, THUMB_TIP)
    index_straight = finger_is_straight(landmarks, INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_TIP)
    middle_straight = finger_is_straight(landmarks, MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP, MIDDLE_FINGER_TIP)

   
    index_higher_than_middle = landmarks[INDEX_FINGER_TIP].y < landmarks[MIDDLE_FINGER_TIP].y

    
    ring_curled = finger_is_curled(landmarks, RING_FINGER_MCP, RING_FINGER_PIP, RING_FINGER_TIP)
    pinky_curled = finger_is_curled(landmarks, PINKY_MCP, PINKY_PIP, PINKY_TIP)

    fingers_not_down = (landmarks[INDEX_FINGER_TIP].y < landmarks[INDEX_FINGER_MCP].y and
                       landmarks[MIDDLE_FINGER_TIP].y < landmarks[MIDDLE_FINGER_MCP].y)

    return (thumb_straight and index_straight and middle_straight and
            index_higher_than_middle and ring_curled and pinky_curled and
            fingers_not_down)

def is_L(landmarks):
    
    index_straight = finger_is_straight(landmarks, INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_TIP)
    
    thumb_horizontal = (
        abs(landmarks[THUMB_TIP].y - landmarks[THUMB_MCP].y) < 0.12 and  
        landmarks[THUMB_TIP].x < landmarks[INDEX_FINGER_MCP].x and  
        landmarks[THUMB_TIP].y > landmarks[INDEX_FINGER_MCP].y  
    )
    
   
    middle_closed = get_distance(landmarks[MIDDLE_FINGER_TIP], landmarks[MIDDLE_FINGER_MCP]) < 0.2
    ring_closed = get_distance(landmarks[RING_FINGER_TIP], landmarks[RING_FINGER_MCP]) < 0.2
    pinky_closed = get_distance(landmarks[PINKY_TIP], landmarks[PINKY_MCP]) < 0.2
    
   
    return (
        thumb_horizontal and index_straight and middle_closed and ring_closed and pinky_closed 
    )

def is_M(landmarks):
    
    fingers_straight = all(finger_is_straight(landmarks, mcp, pip, tip) for mcp, pip, tip in [
        (INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_TIP),
        (MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP, MIDDLE_FINGER_TIP),
        (RING_FINGER_MCP, RING_FINGER_PIP, RING_FINGER_TIP),
        (PINKY_MCP, PINKY_PIP, PINKY_TIP)
    ])
    
    fingers_down = all(
        landmarks[tip].y > landmarks[mcp].y + 0.05  
        for tip, mcp in [
            (INDEX_FINGER_TIP, INDEX_FINGER_MCP),
            (MIDDLE_FINGER_TIP, MIDDLE_FINGER_MCP),
            (RING_FINGER_TIP, RING_FINGER_MCP),
            (PINKY_TIP, PINKY_MCP)
        ]
    )
    
    thumb_folded = (
        landmarks[THUMB_TIP].x < landmarks[THUMB_IP].x and
        landmarks[THUMB_IP].x < landmarks[THUMB_MCP].x and
        landmarks[THUMB_TIP].x < landmarks[INDEX_FINGER_MCP].x
    )
    
    # dita separate tra di loro
    tips = [INDEX_FINGER_TIP, MIDDLE_FINGER_TIP, RING_FINGER_TIP, PINKY_TIP]
    fingers_spread = all(
        get_distance(landmarks[tips[i]], landmarks[tips[i+1]]) < 0.15
        for i in range(3)
    )
    
    return (fingers_straight and fingers_down and thumb_folded and 
            fingers_spread )

def is_N(landmarks):
    
    fingers_straight = all(finger_is_straight(landmarks, mcp, pip, tip) for mcp, pip, tip in [
        (INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_TIP),
        (MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP, MIDDLE_FINGER_TIP),
    ])
    
    
    fingers_down = all(
        landmarks[tip].y > landmarks[mcp].y + 0.05  # Devono essere significativamente più in basso
        for tip, mcp in [
            (INDEX_FINGER_TIP, INDEX_FINGER_MCP),
            (MIDDLE_FINGER_TIP, MIDDLE_FINGER_MCP),
        ]
    )
    
    thumb_folded = (
        landmarks[THUMB_TIP].x < landmarks[THUMB_IP].x and
        landmarks[THUMB_IP].x < landmarks[THUMB_MCP].x and
        landmarks[THUMB_TIP].x < landmarks[INDEX_FINGER_MCP].x
    )
    

    other_fingers_curled = all(finger_is_curled(landmarks, mcp, pip, tip) for mcp, pip, tip in [
        (RING_FINGER_MCP, RING_FINGER_PIP, RING_FINGER_TIP),
        (PINKY_MCP, PINKY_PIP, PINKY_TIP)
    ])
    
    return (fingers_straight and fingers_down and thumb_folded and other_fingers_curled)

def is_O(landmarks):
    thumb_tip = landmarks[THUMB_TIP]
    index_tip = landmarks[INDEX_FINGER_TIP]
    middle_tip = landmarks[MIDDLE_FINGER_TIP]
    ring_tip = landmarks[RING_FINGER_TIP]
    pinky_tip = landmarks[PINKY_TIP]
    
    
    fingers_curved = all(0.3 < get_angle(landmarks[mcp], landmarks[pip], landmarks[tip]) < 2.8 
                         for mcp, pip, tip in [
                             (INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_TIP),
                             (MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP, MIDDLE_FINGER_TIP),
                             (RING_FINGER_MCP, RING_FINGER_PIP, RING_FINGER_TIP),
                             (PINKY_MCP, PINKY_PIP, PINKY_TIP)
                         ])
    
   
    thumb_position = landmarks[THUMB_TIP].x < landmarks[INDEX_FINGER_PIP].x
    
  
    finger_tips_y = [index_tip.y, middle_tip.y, ring_tip.y, pinky_tip.y]
    max_y_diff = max(finger_tips_y) - min(finger_tips_y)
    fingers_aligned = max_y_diff < 0.08  
    
    finger_tips_x = [index_tip.x, middle_tip.x, ring_tip.x, pinky_tip.x]
    max_x_diff = max(finger_tips_x) - min(finger_tips_x)
    fingers_grouped = max_x_diff < 0.15  
    
    
    fingers_close_to_thumb = all(get_distance(thumb_tip, finger_tip) < 0.1 for finger_tip in [index_tip, middle_tip, ring_tip, pinky_tip])
    
    
    fingers_touch_thumb = sum(get_distance(thumb_tip, finger_tip) < 0.05 for finger_tip in [index_tip, middle_tip, ring_tip, pinky_tip]) >= 2
    
    not_fully_closed = all(get_distance(landmarks[tip], landmarks[mcp]) > 0.05 for tip, mcp in [
        (INDEX_FINGER_TIP, INDEX_FINGER_MCP),
        (MIDDLE_FINGER_TIP, MIDDLE_FINGER_MCP),
        (RING_FINGER_TIP, RING_FINGER_MCP),
        (PINKY_TIP, PINKY_MCP)
    ])

    return (fingers_curved and thumb_position and fingers_aligned and 
            fingers_grouped and fingers_close_to_thumb and fingers_touch_thumb and 
            not_fully_closed)

def is_P(landmarks):

    index_straight = get_distance(landmarks[INDEX_FINGER_TIP], landmarks[INDEX_FINGER_PIP]) > 0.05
    index_horizontal = abs(landmarks[INDEX_FINGER_TIP].y - landmarks[INDEX_FINGER_MCP].y) < 0.18  # Tolleranza per la posizione
    
    middle_straight = get_distance(landmarks[MIDDLE_FINGER_TIP], landmarks[MIDDLE_FINGER_PIP]) > 0.05
    middle_pointing_down = landmarks[MIDDLE_FINGER_TIP].y > landmarks[MIDDLE_FINGER_MCP].y
    
    thumb_touching_middle = get_distance(landmarks[THUMB_TIP], landmarks[MIDDLE_FINGER_TIP]) < 0.05
    
    ring_curled = get_distance(landmarks[RING_FINGER_TIP], landmarks[RING_FINGER_MCP]) < 0.13
    pinky_curled = get_distance(landmarks[PINKY_TIP], landmarks[PINKY_MCP]) < 0.13
    
    return (index_straight and index_horizontal and middle_straight and middle_pointing_down and thumb_touching_middle and ring_curled and pinky_curled)

def is_Q(landmarks):

    index_slightly_bent = (
        get_angle(
            landmarks[INDEX_FINGER_MCP], 
            landmarks[INDEX_FINGER_PIP], 
            landmarks[INDEX_FINGER_TIP]
        ) < 2.5  # Angolo più permissivo per consentire una leggera piegatura
    )
    
    index_pointing_down = landmarks[INDEX_FINGER_TIP].y > landmarks[INDEX_FINGER_MCP].y
    
   
    thumb_straight_down = (
        finger_is_straight(landmarks, THUMB_MCP, THUMB_IP, THUMB_TIP) and
        landmarks[THUMB_TIP].y > landmarks[THUMB_MCP].y
    )
    
    
    other_fingers_curled_down = all(
        landmarks[tip].y > landmarks[mcp].y and
        get_distance(landmarks[tip], landmarks[mcp]) < 0.3
        for tip, mcp in [
            (MIDDLE_FINGER_TIP, MIDDLE_FINGER_MCP),
            (RING_FINGER_TIP, RING_FINGER_MCP),
            (PINKY_TIP, PINKY_MCP)
        ]
    )
    
    return (index_slightly_bent and index_pointing_down and thumb_straight_down and other_fingers_curled_down)

def is_R(landmarks):
   
    index_straight = finger_is_straight(landmarks, INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_TIP)
    
    middle_straight = finger_is_straight(landmarks, MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP, MIDDLE_FINGER_TIP)
    
    index_middle_crossed = get_distance(landmarks[INDEX_FINGER_TIP], landmarks[MIDDLE_FINGER_TIP]) < 0.04

    ring_curled = finger_is_curled(landmarks, RING_FINGER_MCP, RING_FINGER_PIP, RING_FINGER_TIP)
    pinky_curled = finger_is_curled(landmarks, PINKY_MCP, PINKY_PIP, PINKY_TIP)

    
    index_middle_tolerance = abs(landmarks[INDEX_FINGER_TIP].x - landmarks[MIDDLE_FINGER_TIP].x) < 0.05

    return (index_straight and middle_straight and index_middle_crossed and 
            ring_curled and pinky_curled and index_middle_tolerance)

def is_S(landmarks):
    
    thumb_horizontal = (
        abs(landmarks[THUMB_TIP].y - landmarks[THUMB_MCP].y) < 0.1 and  
        landmarks[THUMB_TIP].x < landmarks[INDEX_FINGER_MCP].x and  
        landmarks[THUMB_TIP].y > landmarks[INDEX_FINGER_MCP].y  
    )

    
    fingers_closed = all(finger_is_closed(landmarks, tip, mcp) for tip, mcp in [
        (INDEX_FINGER_TIP, INDEX_FINGER_MCP),
        (MIDDLE_FINGER_TIP, MIDDLE_FINGER_MCP),
        (RING_FINGER_TIP, RING_FINGER_MCP),
        (PINKY_TIP, PINKY_MCP)
    ])

    return thumb_horizontal and fingers_closed

def is_T(landmarks):

    thumb_straight_threshold = 0.1  # Imposta una soglia per la tolleranza
    thumb_length = get_distance(landmarks[THUMB_MCP], landmarks[THUMB_TIP])
    
  
    thumb_is_straight = abs(get_distance(landmarks[THUMB_IP], landmarks[THUMB_TIP]) - thumb_length) < thumb_straight_threshold

    index_is_straight = finger_is_straight(landmarks, INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_TIP)

    index_is_horizontal = abs(landmarks[INDEX_FINGER_TIP].y - landmarks[INDEX_FINGER_MCP].y) < 0.15 and (
        landmarks[INDEX_FINGER_TIP].x < landmarks[INDEX_FINGER_MCP].x
    )

    fingers_closed = all(finger_is_closed(landmarks, tip, mcp) for tip, mcp in [
        (MIDDLE_FINGER_TIP, MIDDLE_FINGER_MCP),
        (RING_FINGER_TIP, RING_FINGER_MCP),
        (PINKY_TIP, PINKY_MCP)
    ])

    return (thumb_is_straight and index_is_straight and index_is_horizontal and fingers_closed)

def is_U(landmarks):
    
    index_straight = finger_is_straight(landmarks, INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_TIP)
    middle_straight = finger_is_straight(landmarks, MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP, MIDDLE_FINGER_TIP)

    
    ring_closed = finger_is_curled(landmarks, RING_FINGER_MCP, RING_FINGER_PIP, RING_FINGER_TIP)
    thumb_on_ring = get_distance(landmarks[THUMB_TIP], landmarks[RING_FINGER_MCP]) < 0.1
    
    
    index_middle_tolerance = get_distance(landmarks[INDEX_FINGER_TIP], landmarks[MIDDLE_FINGER_TIP]) < 0.05

    return index_straight and middle_straight and ring_closed and thumb_on_ring and index_middle_tolerance

def is_V(landmarks): 
    index_tip = landmarks[INDEX_FINGER_TIP]
    middle_tip = landmarks[MIDDLE_FINGER_TIP]

    index_straight = finger_is_straight (landmarks , INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_TIP)
    middle_straight = finger_is_straight (landmarks, MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP, MIDDLE_FINGER_TIP)
    ring_curled = finger_is_curled (landmarks, RING_FINGER_MCP, RING_FINGER_PIP, RING_FINGER_TIP)

    thumb_on_ring = get_distance(landmarks[THUMB_TIP], landmarks[RING_FINGER_MCP]) < 0.1

    fingers_spread = 0.1 < get_distance(index_tip, middle_tip) < 0.9

    return (index_straight and middle_straight  and thumb_on_ring and fingers_spread and ring_curled)

def is_W(landmarks): 

    index_tip = landmarks[INDEX_FINGER_TIP]
    middle_tip = landmarks[MIDDLE_FINGER_TIP]
    ring_tip = landmarks[RING_FINGER_TIP]

    index_straight = finger_is_straight (landmarks , INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_TIP)
    middle_straight = finger_is_straight (landmarks, MIDDLE_FINGER_MCP, MIDDLE_FINGER_PIP, MIDDLE_FINGER_TIP)
    ring_straight = finger_is_straight(landmarks, RING_FINGER_MCP, RING_FINGER_PIP, RING_FINGER_TIP)
    
    thumb_on_pinky = get_distance(landmarks[THUMB_TIP], landmarks[PINKY_MCP]) < 0.1

    fingers_straight = (get_distance(index_tip, middle_tip) > 0.09)
    fingers_straight_ring= (get_distance(middle_tip, ring_tip) > 0.07)
    
    return (index_straight and middle_straight and ring_straight and thumb_on_pinky and fingers_straight and fingers_straight_ring)

def is_X(landmarks):
    
    index_curved = finger_is_curled(landmarks, INDEX_FINGER_MCP, INDEX_FINGER_PIP, INDEX_FINGER_TIP)

    other_fingers_closed = all(finger_is_closed(landmarks, tip, mcp) for tip, mcp in [
        (MIDDLE_FINGER_TIP, MIDDLE_FINGER_MCP),
        (RING_FINGER_TIP, RING_FINGER_MCP),
        (PINKY_TIP, PINKY_MCP)
    ])

    index_height_threshold = 0.13
    index_height = landmarks[INDEX_FINGER_TIP].y - landmarks[INDEX_FINGER_MCP].y  # Altezza dell'indice
    index_is_at_correct_height = abs(index_height) < index_height_threshold 

    thumb_touch_middle = get_distance(landmarks[THUMB_TIP], landmarks[MIDDLE_FINGER_TIP]) < 0.1

    return index_curved and other_fingers_closed and thumb_touch_middle and index_is_at_correct_height

def is_Y(landmarks):
   
    thumb_extended = get_distance(landmarks[THUMB_TIP], landmarks[THUMB_MCP]) > 0.05
    pinky_extended = get_distance(landmarks[PINKY_TIP], landmarks[PINKY_PIP]) > 0.05
    
   
    index_curled = get_distance(landmarks[INDEX_FINGER_TIP], landmarks[INDEX_FINGER_MCP]) < 0.1
    middle_curled = get_distance(landmarks[MIDDLE_FINGER_TIP], landmarks[MIDDLE_FINGER_MCP]) < 0.1
    ring_curled = get_distance(landmarks[RING_FINGER_TIP], landmarks[RING_FINGER_MCP]) < 0.1
    
 
    thumb_up = landmarks[THUMB_TIP].y < landmarks[THUMB_IP].y
    pinky_up = landmarks[PINKY_TIP].y < landmarks[PINKY_PIP].y
    
  
    thumb_pinky_spread = get_distance(landmarks[THUMB_TIP], landmarks[PINKY_TIP]) > 0.2
    
   
    thumb_correct_position = (landmarks[THUMB_TIP].x > landmarks[INDEX_FINGER_MCP].x)
    
    return (thumb_extended and pinky_extended and 
            index_curled and middle_curled and ring_curled and
            thumb_up and pinky_up and 
            thumb_pinky_spread and thumb_correct_position)

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
    elif is_G(landmarks):
        return 'G'
    elif is_H(landmarks):
        return 'H'
    elif is_I(landmarks):
        return 'I'
    elif is_J(landmarks):
        return 'J'
    elif is_K(landmarks):
        return 'K'
    elif is_L(landmarks):
        return 'L'
    elif is_M(landmarks):
        return 'M'
    elif is_N(landmarks):
        return 'N'
    elif is_O(landmarks):
        return 'O'
    elif is_P(landmarks):
        return 'P'
    elif is_Q(landmarks):
        return 'Q'
    elif is_R(landmarks):
        return 'R'
    elif is_S(landmarks):
        return 'S'
    elif is_T(landmarks):
        return 'T'
    elif is_U(landmarks):
        return 'U'
    elif is_V(landmarks):
        return 'V'
    elif is_W(landmarks):
        return 'W'
    elif is_X(landmarks):
        return 'X'
    elif is_Y(landmarks):
        return 'Y'
    else:
        return ''