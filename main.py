import face_recognition
import cv2
import numpy
import random
from playsound import playsound
import asyncio
import threading

video_capture = cv2.VideoCapture(0)

known_face_encodings = []
known_face_names = []

known_face_list = [
    ["images/y.jpg", "FaceID: Y"],
    ["images/r.jpg", "FaceID: R"]
]

black_list = ["FaceID: R"]

for face in known_face_list:
    face_image = face_recognition.load_image_file(face[0])
    face_encoding = face_recognition.face_encodings(face_image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(face[1])

face_id = 1
face_names = []
names_body_temp = {}
names_counter = {}
old_len = 0
isAlert = False

seijoWav = "sounds/seijo.wav"
blackWav = "sounds/black.wav"

async def soundGoGo ():
    if not isAlert:
        playsound(blackWav)
    else:
        playsound(seijoWav)


while True:
    print(face_names)
    old_len = len(face_names)
    ret, frame = video_capture.read()

    rgb_frame = numpy.ascontiguousarray(frame[:, :, ::-1])

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_names = []
    isAlert = False
    
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.45)
        name = "FaceID:{}".format(face_id)
        
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = numpy.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        else:
            known_face_encodings.append(face_encoding)
            known_face_names.append(name)
            face_id = face_id + 1
        face_names.append(name)

        if name in names_body_temp:
            temp = names_body_temp[name]
        else:
            if name in black_list:
                names_body_temp[name] = round(random.randint(375, 390) / 10, 1)
            else:
                names_body_temp[name] = round(random.randint(360, 369) / 10, 1)
            temp = names_body_temp[name]
            
        if name in black_list:
            isAlert = True
            
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        cv2.putText(frame, str(temp), (left + 6, top -6), font, 1.0, (255, 255, 255), 1)

    if old_len < len(face_names):
        thread = threading.Thread(target=lambda: asyncio.run(soundGoGo()))
        thread.start()
        
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
