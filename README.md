# Face Recognition Attendance System 🎯

## About
An automatic attendance system that uses 
face recognition to mark students attendance 
in real-time.

## Features
- Real-time face detection
- Automatic attendance marking
- Saves attendance in CSV file
- High accuracy using DeepFace

## Tech Stack
- Python
- DeepFace
- OpenCV
- NumPy
- Pandas

## How it Works
1. 100 photos of each student are captured
2. Face embeddings are created using DeepFace
3. Real-time camera matches the face
4. Attendance is saved in CSV file

## How to Run
1. Install libraries:
   pip install deepface opencv-python pandas numpy

2. Collect Dataset:
   python face_recognization.py

3. Build Embeddings:
   python build_embeddings.py

4. Start Attendance System:
   python attendance.py

## Project Structure
face_recognization.py  - Dataset collection
build_embeddings.py    - Face embeddings generator
attendance.py          - Attendance marking system
