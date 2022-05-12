import dlib
import cv2
import numpy as np

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
        
        
        return temp_img
    
    
if __name__ == '__main__':
    cap = cv2.VideoCapture(0) 
    folder = "dataset/test/middle/"
    dr = input('start?')
    counter = 0
    while counter < 30:
        ret,frame = cap.read()
        detected_eye = getEye(frame)
        print(f"{counter+1} image")
        
        cv2.imwrite(folder+str(counter)+".png",detected_eye)
        counter += 1
        cv2.imshow('eyes', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    cap.release()
    cv2.destroyAllWindows()
