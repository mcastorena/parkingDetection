import yaml
import numpy as np
import cv2
import json
from parkingSpaceBoundary import parkingSpaceBoundary
from os.path import isfile
from requests import post

class parkingDetector:
    def __init__(self, videoFile, classifierXML, ymlFile, jsonFile):
        self.video = videoFile
        self.parkingSpaceYML = ymlFile
        self.jsonFilePath = jsonFile
        self.carCascade = cv2.CascadeClassifier(classifierXML)

        self.parkingOverlay = True
        self.parkingSpaceData = []
        self.parkingData = {}
        self.parkingCountours = []
        self.parkingSpaceBoundingRectangles = []
        self.parkingMask = []
        self.parkingDataMotion = []
        self.parkingStatus = []
        self.parkingBuffer = []

        self.cap = cv2.VideoCapture(self.video)                             #open video

        self.erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))   # Initialize morphological kernels
        self.dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 19))    # erode is used to remove white noise (but also shrinks our object) and dilate to expand our shrunk objects

        self.parkingThreshold = 3.5
        self.secsToWait = 2

        self.URL = ""                                                        # Server URL and port
        self.authenticationToken = ""                                        # Authentication Token file location
        self.cameraNum = 10                                                  # Camera number used to identify where the info is coming from


    def run(self):
        self.openYML()                                          # Open ymlFile and save parking space boundary information

        self.erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))   # Initialize morphological kernels
        self.dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 19))

        # Parking detection begins

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 1700)  # jump to frame number specified

        while (self.cap.isOpened()):
            currentPosition = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0  # Current position of the video file in seconds
            currentFrame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)  # Index of the frame to be decoded/captured next

            success, initialFrame = self.cap.read()                                  # Capture frame
            if success:
                frame = cv2.resize(initialFrame, None, fx=0.6, fy=0.6)          # If frame capture is successful resize frame
            else:
                print("Video ended")
                break

            # Background subtraction using grayscale gaussian blur
            frameGBlur = cv2.GaussianBlur(frame.copy(), (5, 5), 3)
            grayGBlur = cv2.cvtColor(frameGBlur, cv2.COLOR_BGR2GRAY)

            frameOut = frame.copy()                                             # Frame used to draw over and display


            for index, park in enumerate(self.parkingSpaceData):
                points = np.array(park['points'])  # get coordinates for that parking spaces bounding corners
                rect = self.parkingSpaceBoundingRectangles[index]  # load rectangle for parking space at index = index
                roi_gray = grayGBlur[rect[1]:(rect[1] + rect[3]), rect[0]:(rect[0] + rect[2])]  # Crop ROI from frame using boundary points
                status = self.runClassifier(roi_gray)

                # If detected a change in parking status, save the current time
                if status != self.parkingStatus[index] and self.parkingBuffer[index] == None:
                    self.parkingBuffer[index] = currentPosition


                # If status is still different than the one saved and counter is open
                elif status != self.parkingStatus[index] and self.parkingBuffer[index] != None:
                    if currentPosition - self.parkingBuffer[index] > self.secsToWait:
                        self.parkingStatus[index] = status
                        self.parkingBuffer[index] = None

                        # Loads dictionary with parkingStatus info, turns dictionary into JSON object and writes to JSON file
                        for ind, val in enumerate(self.parkingStatus):          # Copy parkingStatus contents into parkingData (dictionary)
                            self.parkingData[ind] = str(val)
                        print(self.parkingData)
                        jsonData = json.dumps(self.parkingData)  # Dictionary to json
                        with open(self.jsonFilePath, 'w') as outfile:  # Write to JSON file
                            json.dump(jsonData, outfile)

                        # Send JSON file to server
                        # self.postStatus()


                # If status is still same and counter is open clear buffer
                elif status == self.parkingStatus[index] and self.parkingBuffer[index] != None:
                    self.parkingBuffer[index] = None

            # Overlays parking space boundary lines and ID numbers on the frames displayed
            # Works well for testing but unnecessarily uses up resources for cameras in actual lots
            if self.parkingOverlay:
                for ind, space in enumerate(self.parkingSpaceData):
                    if self.parkingStatus[ind]:
                        color = (0, 255, 0)                                 # If status is True (vacant) then color = green
                    else:
                        color = (0, 0, 255)                                 # Else color is red

                    points = np.array(space['points'])                      # Get space's boundary points
                    cv2.drawContours(frameOut, [points], contourIdx=-1, color=color, thickness=2, lineType=cv2.LINE_8)          # Draw boundary box

                    moments = cv2.moments(points)
                    centroid = (int(moments['m10'] / moments['m00']) - 3, int(moments['m01'] / moments['m00']) + 3)
                    # Putting numbers on marked regions
                    cv2.putText(frameOut, str(space['id']), (centroid[0] + 1, centroid[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frameOut, str(space['id']), (centroid[0] - 1, centroid[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frameOut, str(space['id']), (centroid[0] + 1, centroid[1] - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frameOut, str(space['id']), (centroid[0] - 1, centroid[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(frameOut, str(space['id']), centroid, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)



            # Display video
            cv2.imshow('frame', frameOut)
            k = cv2.waitKey(1)
            if k == ord('q'):
                break
            elif k == ord('c'):
                cv2.imwrite('frame%d.jpg' % currentFrame, frameOut)
            elif k == ord('j'):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, currentFrame + 500)  # jump 500 frames forward
            elif k == ord('u'):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, currentFrame - 500)  # jump 500 frames back
            if cv2.waitKey(33) == 27:
                break

        cv2.waitKey(0)
        self.cap.release()
        cv2.destroyAllWindows()


    def runClassifier(self, img):
        cars = self.carCascade.detectMultiScale(img, 1.1, 1)
        if cars == ():
            return True
        else:
            return False

    def postStatus(self):
        print("Sending status data...")
        head = {'Authorization': self.authenticationToken}
        body = {'Camera': self.cameraNum, 'status':self.jsonFilePath}                       # Need to parse jsonFile first, will do later
        try:
            msgResponse = post(self.URL, headers= head, data= body)                         # Post message
        except:
            raise ValueError('Status could not be posted')                                  # If post message fails catch error and print message


    def openYML(self):
        # Read YAML data (parking space polygons)
        if isfile(self.parkingSpaceYML):  # If yml file exists open it and load parking space data
            with open(self.parkingSpaceYML, 'r') as stream:
                self.parkingSpaceData = yaml.load(stream)
        else:  # Else create yml file then load it
            success, image = self.cap.read()  # Capture frame to mark parking spaces
            if success:  # If frame can be captured
                ymlImg = cv2.resize(image, None, fx=0.6, fy=0.6)  # Resize captured frame
                mySpace = parkingSpaceBoundary(ymlImg, self.parkingSpaceYML)  # Create parkingSpaceBoundary object and pass it resized frame and destination yml file path
                mySpace.markSpaces()  # Run function to mark parking space boundaries
                del mySpace  # Delete object once done

                with open(self.parkingSpaceYML, 'r') as stream:  # Open new yml file and load it
                    self.parkingSpaceData = yaml.load(stream)
            else:  # If frame cannot be captured then print error message and exit
                print("Video could not be opened, parking boundaries cannot be established")
                raise SystemExit


        if self.parkingSpaceData != None:
            for park in self.parkingSpaceData:
                points = np.array(park['points'])
                rect = cv2.boundingRect(points)
                points_shifted = points.copy()
                points_shifted[:, 0] = points[:, 0] - rect[0]  # shift contour to region of interest
                points_shifted[:, 1] = points[:, 1] - rect[1]
                self.parkingCountours.append(points)
                self.parkingSpaceBoundingRectangles.append(rect)
                mask = cv2.drawContours(np.zeros((rect[3], rect[2]), dtype=np.uint8), [points_shifted], contourIdx=-1,
                                        color=255, thickness=-1, lineType=cv2.LINE_8)
                mask = mask == 255

                self.parkingMask.append(mask)

                self.parkingStatus = [False] * len(self.parkingSpaceData)            # Initialize parkingStatus with length of parking spaces read and initialize all parking sports as occupied
                self.parkingBuffer = [None] * len(self.parkingSpaceData)             # Initialize parkingBuffer
