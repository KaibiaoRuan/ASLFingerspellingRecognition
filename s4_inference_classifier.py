import pickle

import cv2
import mediapipe as mp
import numpy as np

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(2)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
#labels_dict = {0: ',', 1: ' ', 2: 'A', 3: 'C', 4: 'I', 5: 'L', 6: 'O', 7: 'P', 8: 'T', 9: 'U', 10: 'S'}
#character_counts = {',': 0, ' ': 0, 'A': 0, 'C': 0, 'I': 0, 'L': 0, 'O': 0, 'P': 0, 'T': 0, 'U': 0, 'S': 0}
labels_dict = {0: ',', 1: ' ', 2: 'N', 3: 'E', 4: 'W', 5: 'S'}
character_counts = {',': 0, ' ': 0, 'N': 0, 'E': 0, 'W': 0, 'S': 0}
numLetter = 0
repeat = 50
while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])
        
        predicted_character = labels_dict[int(prediction[0])]

        if predicted_character in character_counts:
            character_counts[predicted_character] += 1

        for char, count in character_counts.items():
            if count > repeat:
                print(char, end = "", flush=True)
                character_counts = dict.fromkeys(character_counts, 0)
                numLetter += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3,
                    cv2.LINE_AA)
    else:
        if numLetter > 40:
            print(".")
            numLetter = 0


    cv2.imshow('frame', frame)
    cv2.waitKey(10)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
