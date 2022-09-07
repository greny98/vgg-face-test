import tensorflow as tf
import cv2

from vgg_face.model import create_model
from vgg_face.utils import identify, embedding_dir


if __name__ == '__main__':
    # model = create_model("models/vgg_face_weights.h5")
    model = tf.saved_model.load("models/vgg_face")
    embeddings = embedding_dir(model, "data")
    # img = cv2.imread("data/quang_anh/img1.jpeg")
    # img = cv2.imread("data/test.jpeg")
    # identify(model, img, embeddings)
    # # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    # cv2.imshow("frame", img)
    # k = cv2.waitKey(0) & 0xFF
    # while k != ord('q'):
    #     k = cv2.waitKey(0) & 0xFF
    # cv2.destroyAllWindows()

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Read error")
            break
        identify(model, frame, embeddings)
        # Show
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
