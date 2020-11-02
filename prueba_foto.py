import cv2
import numpy as np
import math

def detect_eyes(img, classifier):
    global bounding_lefteye, bounding_righteye
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eyes = classifier.detectMultiScale(gray_frame, 1.3, 5) # detectar ojos
    width = np.size(img, 1) # obtener el ancho del marco de la cara
    height = np.size(img, 0) # obtener la altura del marco de la cara
    left_eye = None
    right_eye = None
    for (x, y, w, h) in eyes:
        if y < height / 2:
            eyecenter = x + w / 2  # obtener el centro del ojo
            if eyecenter < width * 0.5:
                print('left_eye',x, y, w, h) # se imprime la posición en x,y del ojo izquierdo 
                left_eye = img[y:y + h, x:x + w]
                cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2) # se dibuja el marco del ojo izquierdo
            else:
                print('right_eye',x, y, w, h) # se imprime la posición en x,y del ojo derecho
                right_eye = img[y:y + h, x:x + w]
                cv2.rectangle(img, (x, y), (x + w, y + h), 255, 2) # se dibuja el marco del ojo derecho
    return left_eye, right_eye

def detect_faces(img, classifier):      # función para detectar la cara
    global biggest
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = classifier.detectMultiScale(gray_frame, 1.3, 5)
    if len(coords) > 1:
        biggest = (0, 0, 0, 0)
        for i in coords:
            if i[3] > biggest[3]:
                biggest = i
        biggest = np.array([i], np.int32)
    elif len(coords) == 1:
        biggest = coords
    else:
        return None
    for (x, y, w, h) in biggest:
        print('face', x, y, w, h)
        frame = img[y:y + h, x:x + w]
        cv2.rectangle(img, (x,y), (x+w,y+h), 255, 2)
    return frame

def cut_eyebrows(img):
    height, width = img.shape[:2]
    eyebrow_h = int(height / 4)
    img = img[eyebrow_h:height, 0:width]  # cut eyebrows out (15 px)
    return img

def blob_process(img, threshold, detector):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
    img = cv2.erode(img, None, iterations=2) #1
    img = cv2.dilate(img, None, iterations=4) #2
    img = cv2.medianBlur(img, 5) #3
    keypoints = detector.detect(img)
    return keypoints



def main():
    photo = cv2.imread('/Users/lauestupinan/Desktop/T/34/foto2.jpg')    # se ingresa la ruta de la imagen que se quiere procesar
    face_frame = detect_faces(photo, face_cascade)
    if face_frame is not None:
        eyes = detect_eyes(face_frame, eye_cascade)
        j=0
        nombre = ['ojo1', 'ojo2', 'ojo3', 'ojo4']
        for eye in eyes:
            if eye is not None:

                eye = cut_eyebrows(eye)
                gray_frame = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
                threshold = 60
                ret, Ibw_manual = cv2.threshold(gray_frame, threshold, 255, cv2.THRESH_BINARY)
                opening = cv2.morphologyEx(Ibw_manual, cv2.MORPH_OPEN, (5, 5))
                closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, (5, 5))
                cv2.imshow(nombre[j], closing)
                contours, hierarchy = cv2.findContours(closing, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                i=0
                for idx, cont in enumerate(contours):
                    #print(i)
                    M = cv2.moments(contours[idx])
                    area = int(M['m00'])
                    #print(area)
                    if area > 100 and area <2000:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        radio = int(math.sqrt(area/math.pi))
                        print('center_eyes', cx, cy, radio)
                        cv2.circle(eye, (cx, cy), radio, (0, 255, 0), 2)
                        cv2.circle(eye, (cx, cy), 1, (0, 0, 255), 3)
                    i+=1
            j+=1
    cv2.imshow('image', photo)        # se muestra la imagen procesada
    cv2.imshow('image1', face_frame)  # se muestra el marco de la cara con los ojos detectados
    cv2.imshow('image2', eyes[0])     # se muestra el marco del ojo izquierdo
    cv2.imshow('image3', eyes[1])     # se muestra el marco del ojo derecho
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def nothing(x):
    pass

if __name__ == '__main__':
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    main()
