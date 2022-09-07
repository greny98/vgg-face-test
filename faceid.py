import numpy as np
import cv2
import dlib
import face_recognition
import os
import pickle
import time


detector = dlib.get_frontal_face_detector()

def embedding_dir(dir: str) -> dict:
    """
    Embedding all employees. In <dir>, each employee have folder named as his/her name.
    :param dir: Directory containing photos of company employees
    :return: Dictionary with {name: [embeddings]}
    """
    embeddings = {}
    if os.path.exists("embeddings.pickle"):
        print("Reading")
        with open('embeddings.pickle', 'rb') as handle:
            embeddings = pickle.load(handle)
            return embeddings

    for name in os.listdir(dir):
        if name != ".DS_Store":
            name_path = os.path.join(dir, name)
            vectors = []
            for filename in os.listdir(name_path):
                if filename != ".DS_Store":
                    img = cv2.imread(os.path.join(name_path, filename))
                    embs = face_recognition.face_encodings(img)
                    if len(embs) > 0:
                        vectors.append(embs)
            embeddings[name] = vectors
    with open('embeddings.pickle', 'wb') as handle:
        pickle.dump(embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return embeddings

def draw(img: np.ndarray, loc, name_draw):
    t, r, b, l = loc
    cv2.rectangle(img, (l, t), (r, b), (0, 255, 0), 2)
    cv2.putText(img, name_draw, (l, t), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def show(title, img: np.ndarray):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def face_id(img: np.ndarray):
    frame_clone = img.copy()
    frame_clone = cv2.resize(frame_clone, (0, 0), fx=0.5, fy=0.5)
    face_locs = face_recognition.face_locations(frame_clone)
    if len(face_locs) == 0:
        return

    face_locs = [(t * 2 + 3, r * 2 - 3, b * 2 - 3, l * 2) for t, r, b, l in face_locs]
    for loc in face_locs:
        inp_emb = face_recognition.face_encodings(img, [loc])
        name_draw = "Unknown"
        min_wrong = 100
        for name, embs in embeddings.items():
            sim = 0
            for emb in embs:
                sim += face_recognition.compare_faces(emb, inp_emb[0], tolerance=0.57)[0]
            wrong = len(embs) - sim
            if wrong < min_wrong:
                name_draw = name
                min_wrong = wrong
        name_draw = name_draw if min_wrong <= 2 else "Unknown"
        print(min_wrong)
        draw(img, loc, name_draw)

if __name__ == '__main__':
    start = time.time_ns()
    embeddings = embedding_dir("data")
    end = time.time_ns()
    print(end - start)

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Read error")
            break
        face_id(frame)
        # Show
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    # img = cv2.imread("test.jpeg")
    # face_id(img)
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
