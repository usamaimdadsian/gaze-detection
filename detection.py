import dlib
import cv2
import numpy as np
from keras.models import load_model


model = load_model('saved_model.h5')
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def eyes_contour_points(facial_landmarks):
    left_eye = []
    right_eye = []
    for n in range(36, 42):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        left_eye.append([x, y])
    for n in range(42, 48):
        x = facial_landmarks.part(n).x
        y = facial_landmarks.part(n).y
        right_eye.append([x, y])
    left_eye = np.array(left_eye, np.int32)
    right_eye = np.array(right_eye, np.int32)
    return left_eye, right_eye

def getEye(rgb_img):
    gray_img = cv2.cvtColor(rgb_img,cv2.COLOR_BGR2GRAY)
    rects = detector(gray_img,1)
    for rect in rects:
        landmarks = predictor(rgb_img,rect)
        left_eye, right_eye = eyes_contour_points(landmarks)
        
        padd = 2
        y1=min(left_eye[:,0])-padd
        y2=max(left_eye[:,0])+padd

        x1=min(left_eye[:,1])-padd
        x2=max(left_eye[:,1])+padd
        
        temp_img = rgb_img[x1:x2,y1:y2]
        
        resized_img = cv2.resize(temp_img,(32,32))
        norm_img = resized_img / 255.0
        
        return norm_img


if __name__ == '__main__':
    cap = cv2.VideoCapture(0) 
    classes = ['left','middle','right','up']
    while True:
        ret,frame = cap.read()
        detected_eye = getEye(frame)
        
        detected_eye = np.expand_dims(detected_eye,axis=0)
        detection = classes[np.argmax(model.predict(detected_eye),axis=1)[0]]

        print(detection,type(detection))
        h,w,_ = frame.shape
        frame = cv2.circle(frame,(int(w/2),int(h/2)) , 5, (0,0,255), 2)
        cv2.putText(frame, detection, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1, cv2.LINE_AA)
        
        cv2.imshow('eyes', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()


    
