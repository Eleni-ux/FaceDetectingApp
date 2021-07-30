import cv2
from random import randrange


def detect_human_face_in_img(img_name):
    # train the model
    train_face_img = cv2.CascadeClassifier(
        'haarcascade_frontalface_default.xml')
    # get the imgage
    img = cv2.imread(img_name)
    # change the color of the image to gray
    change_to_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detecting face in the image and get the coordinates
    get_coordinates = train_face_img.detectMultiScale(change_to_gray)
    # draw rectangle around the face
    for (x, y, z, w) in get_coordinates:
        cv2.rectangle(img, (x, y), (x+z, y+w), (randrange(256),
                                                randrange(256), randrange(256)), 2)
    cv2.imshow('face detector app', img)
    cv2.waitKey()


detect_human_face_in_img('image1.jpg')
