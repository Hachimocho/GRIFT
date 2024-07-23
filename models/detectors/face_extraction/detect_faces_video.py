from imutils.video import VideoStream
import imutils
from imutils import face_utils
import numpy as np
import argparse
import time
import cv2
from math import floor
import dlib
from tqdm import tqdm
 
class Preprocessor:
    def __init__(self, video, prototxt, model, confidence=0.7, face_dims=(256,256)):
        self.video_path = video
        print("VideoPath", video)
        self.reader = cv2.VideoCapture(video)
        self.legth = int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT))
        self.last_frame = self.next_frame(True)
        self.faces = []
        self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
        self.confidence = confidence
        self.face_dims = face_dims
    

    def next_frame(self, return_frame=False):
        success, frame = self.reader.read()
        frame = imutils.resize(frame, width=1000)
        if return_frame:
            return frame
        else:
            self.last_frame = frame
            return success
    

    def next_detection(self, frame):
        # create a blob for the detector
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (103.93, 116.77, 123.68))
        
        # pass the blob to the detector network and obtain the detections and predictions
        self.net.setInput(blob)
        return self.net.forward()

    @staticmethod
    def get_image_slice(img, y0, y1, x0, x1):
        '''Get values outside the domain of an image'''
        m, n = img.shape[:2]
        padding = max(-y0, y1-m, -x0, x1-n, 0)
        padded_img = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), 'reflect')
        return padded_img[(padding + y0):(padding + y1),
                        (padding + x0):(padding + x1)]

    @staticmethod
    def expand_slice(startX, startY, endX, endY, max_y, max_x, multiplier=1.2):
        w = endX - startX
        h = endY - startY
        cx = startX + (w/2)
        cy = startY + (h/2)

        sq_d = max([h,w]) * multiplier

        n_startX = cx - (sq_d/2)
        n_endX = n_startX + sq_d

        n_startY = cy - (sq_d/2)
        n_endY = n_startY + sq_d

        if n_startX < 0:
            n_startX = 0
            n_endX = n_startX + sq_d
        if n_startY < 0:
            n_startY = 0
            n_endY = n_startY + sq_d
        if n_endX > max_x:
            n_endX = max_x
            n_startX = n_endX - sq_d
        if n_endY > max_y:
            n_endY = max_x
            n_startY = n_endY - sq_d
        
        return floor(n_startX), floor(n_startY), floor(n_endX), floor(n_endY)
        # , floor(cx), floor(cy)

    def add_frame(self, startX, startY, endX, endY):
        # extract face from last frame
        patch = self.last_frame[startY:endY, startX:endX, :]
        if patch.shape[0] == 0 or patch.shape[1] == 0:
            return
        # resize and push to face collector
        
        cv2.imshow("Frame", patch)
        key = cv2.waitKey(1) & 0xFF
        
        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            exit()
        
        self.faces.append(cv2.resize(patch, self.face_dims))

    def run(self):
        
        for i in tqdm(range(self.legth-1)):
            frame = self.last_frame
            height, width = frame.shape[:2]
            
            detections = self.next_detection(frame)
            count = 0
            
            # loop over the detected faces
            for j in range(0, detections.shape[2]):

                # extract the confidence/probability associated with the prediction
                confidence = detections[0,0,j,2]

                # filter out predictions below threshold
                if confidence < self.confidence:
                    continue
                
                count += 1
                
                box = detections[0,0,j,3:7] * np.array([width, height, width, height])
                
                # covnert the coordinates to int and pass to reshape to an expanded square
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY, endX, endY) = self.expand_slice(startX, startY, endX, endY, height, width, multiplier=1.2)

                self.add_frame(startX, startY, endX, endY)
                
            success = self.next_frame()
            if success == False:
                break
            
        cv2.destroyAllWindows()
            


'''
if __name__ == "__main__":
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", help="path to Caffe 'deploy' prototxt file",
        default="deploy.prototxt")
    ap.add_argument("-m", "--model", help="path to Caffe pre-trained model",
        default="res10_300x300_ssd_iter_140000.caffemodel")
    ap.add_argument("-c", "--confidence", help="minimum probability to filter weak detections",
        type=float, default=0.7)
    ap.add_argument("-v", "--video", default="/home/js8365/data/Sandbox/ClassNSeg/predict.mp4",
        help="video path")
    args = vars(ap.parse_args())

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(args["video"])
    success, frame = vs.read()
    count = 0

    while success == True:
        # loop over the frames from the video stream
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = imutils.resize(frame, width=1000)
        frame_p = frame

        # grab the frame dimensions and convert it to a blob
        (f_h, f_w) = frame.shape[:2]    

        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0,(300, 300), (103.93, 116.77, 123.68))
        
        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()
        count = 0    
        
        # loop over the detections
        for i in range(0, detections.shape[2]):
            
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]
            #print(confidence * 100)

    
            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence < args["confidence"]:
                continue
            count += 1 
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([f_w, f_h, f_w, f_h])
            (startX, startY, endX, endY) = box.astype("int")
            
            dlib_rect = dlib.rectangle(startX, startY, endX, endY)
            shape = predictor(frame, dlib_rect)
            shape = face_utils.shape_to_np(shape)

            (startX, startY, endX, endY, cx, cy) = expand_slice(startX, startY, endX, endY, f_h, f_w)

            
            # draw the bounding box of the face along with the associated
            # probability
            text = "{:.2f}%".format(confidence * 100) + ", CV #" + str(count)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame_p, (startX, startY), (endX, endY),(255, 255, 0), 2)
            cv2.putText(frame_p, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 2)

            for (x, y) in shape:
                cv2.circle(frame_p, (x, y), 1, (255, 255, 0), -1)

            # show the output frame
        cv2.imshow("Frame", frame_p)
        key = cv2.waitKey(1) & 0xFF
        
        success, frame = vs.read()
        
        # if the `q` key was pressed, break from the loop
        if (success == False) or (key == ord("q")):
            break
    
    # do a bit of cleanup
    cv2.destroyAllWindows()
'''