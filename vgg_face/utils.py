import os
import pickle
import cv2
import dlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications.imagenet_utils import preprocess_input


detector = dlib.get_frontal_face_detector()

def preprocess_image(faces):
    tensors = []
    for face_img in faces:
        if face_img.shape[0] * face_img.shape[1] != 0:
            img_tensor = tf.convert_to_tensor(face_img, dtype=tf.float32)
            img_tensor = tf.image.resize(img_tensor, [224, 224])
            tensors.append(img_tensor)
    tensors = tf.convert_to_tensor(tensors)
    tensors = preprocess_input(tensors)
    return tensors

def findCosineSim(list_faces, face_check):
    if len(face_check.get_shape()) == 1:
        face_check = tf.expand_dims(face_check, axis=0)
    list_faces = tf.squeeze(list_faces, axis=1)
    dot = tf.matmul(list_faces, tf.transpose(face_check))
    len_list_face = tf.reduce_sum(list_faces * list_faces, axis=1, keepdims=True)
    len_face_check = tf.reduce_sum(face_check * face_check, axis=1, keepdims=True)
    return dot / (tf.sqrt(len_list_face) * tf.sqrt(len_face_check))

def detect_face(img: np.ndarray):
    frame_clone = img.copy()
    frame_clone = cv2.resize(frame_clone, (0, 0), fx=0.5, fy=0.5)
    face_locs = detector(frame_clone, 1)
    faces = []
    locs = []
    for loc in face_locs:
        l, t, r, b = loc.left() * 2, loc.top() * 2, loc.right() * 2, loc.bottom() * 2
        w = r - l
        h = b - t
        if w > h:
            b += (w - h)
        else:
            r += (h - w)

        faces.append(img[t:b, l:r, :])
        locs.append((t, r, b, l))
    return faces, locs

def draw(img: np.ndarray, loc, name_draw):
    t, r, b, l = loc
    color = (0, 255, 0) if name_draw != "Unknown" else (0, 0, 255)
    cv2.rectangle(img, (l, t), (r, b), color, 2)
    cv2.putText(img, name_draw, (l, t), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def embedding_img(model: Model, faces):
    tensor = preprocess_image(faces)
    embed_vector = model(tensor)
    return embed_vector

def embedding_dir(model: Model, dir: str):
    embedding = {}
    saved_file = "embedding.pickle"
    if os.path.exists(saved_file):
        print("Reading")
        with open(saved_file, 'rb') as handle:
            embeddings = pickle.load(handle)
            return embeddings

    for name in os.listdir(dir):
        if name != ".DS_Store":
            name_path = os.path.join(dir, name)
            vectors = []
            for filename in os.listdir(name_path):
                if filename != ".DS_Store":
                    img = cv2.imread(os.path.join(name_path, filename))
                    faces, _ = detect_face(img)
                    emb_vec = embedding_img(model, faces)
                    vectors.append(emb_vec)
            embedding[name] = tf.convert_to_tensor(vectors)
    with open(saved_file, 'wb') as handle:
        pickle.dump(embedding, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return embedding

def identify(model, img, embeddings):
    faces, locs = detect_face(img)
    if len(faces) == 0:
        return
    inp_emb = embedding_img(model, faces)
    # find
    for i in range(inp_emb.shape[0]):
        name_draw = "Unknown"
        max_score = 0
        for name, embs in embeddings.items():
            score = findCosineSim(embs, inp_emb[i])
            score = tf.reduce_mean(score).numpy()
            if score > max_score and score >= 0.75:
                max_score = score
                # name_draw = "{} {:.3f}".format(name, score)
                name_draw = name
        draw(img, locs[i], name_draw)
