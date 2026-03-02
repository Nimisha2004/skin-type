import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("skin_type_model_v2.h5")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

labels = ["oily", "normal", "dry"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        face_resized = cv2.resize(face, (224, 224))
        face_resized = face_resized / 255.0
        face_resized = np.expand_dims(face_resized, axis=0)

        preds = model.predict(face_resized, verbose=0)
        label = labels[np.argmax(preds)]

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Webcam Skin Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
