import cv2
import pickle
import numpy as np
from deepface import DeepFace
import os
import pandas as pd
from datetime import datetime

# ---------- Load embeddings ----------
with open("embeddings.pkl", "rb") as f:
    data = pickle.load(f)

known_embeddings = data["embeddings"]
known_names = data["names"]

# ---------- Create attendance file ----------
attendance_file = "attendance.csv"

if not os.path.exists(attendance_file):

    df = pd.DataFrame(columns=["Name", "Time"])
    df.to_csv(attendance_file, index=False)

    print("attendance.csv created")

# ---------- Start camera ----------
cap = cv2.VideoCapture(0)

print("Camera started. Press Q or ESC to exit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    try:
        result = DeepFace.represent(frame, enforce_detection=False)

        live_embedding = result[0]["embedding"]

        distances = np.linalg.norm(
            np.array(known_embeddings) - live_embedding,
            axis=1
        )

        min_index = np.argmin(distances)

        if distances[min_index] < 15:

            name = known_names[min_index]

            # Show name on screen
            cv2.putText(frame, name, (50,50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,0), 2)

            # ---------- Mark attendance ----------
            now = datetime.now()
            time = now.strftime("%H:%M:%S")

            df = pd.read_csv(attendance_file)

            if name not in df["Name"].values:

                df.loc[len(df)] = [name, time]
                df.to_csv(attendance_file, index=False)

                print("Attendance Marked:", name)

    except:
        pass

    cv2.imshow("Face Attendance System", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q') or key == 27:
        print("Program closed")
        break

cap.release()
cv2.destroyAllWindows() 