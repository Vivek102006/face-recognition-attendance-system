import os
import pickle
from deepface import DeepFace

dataset_path = "dataset"

embeddings = []
names = []

for person in os.listdir(dataset_path):

    person_folder = os.path.join(dataset_path, person)

    for img in os.listdir(person_folder):

        img_path = os.path.join(person_folder, img)

        try:
            result = DeepFace.represent(img_path)

            embeddings.append(result[0]["embedding"])
            names.append(person)

        except:
            pass

data = {
    "embeddings": embeddings,
    "names": names
}

with open("embeddings.pkl", "wb") as f:
    pickle.dump(data, f)

print("Embeddings saved")