import cv2
import yaml

class parkingSpaceBoundary:
    def __init__(self, img, file):
        self.image = img
        self.filePath = file
        self.parkingSpace = []
        self.id = 1

    data = []

    def dumpYML(self):
        with open(self.filePath, "a") as yml:
            yaml.dump(self.data, yml)


    def defineBoundaries(self, event, x, y, flags, param):
        currentSpace = {'id': id, 'points': []}                                      # Initialize dictionary for 1st parking space
        if event == cv2.EVENT_LBUTTONDBLCLK:                                        # If a point on the image is double left clicked
            self.parkingSpace.append((x, y))                                             # Append the point to parkingSpace
        if len(self.parkingSpace) == 4:                                                  # If 4 points have been appended
            cv2.line(self.image, self.parkingSpace[0], self.parkingSpace[1], (0, 255, 0), 1)         # Draw the space on the image
            cv2.line(self.image, self.parkingSpace[1], self.parkingSpace[2], (0, 255, 0), 1)
            cv2.line(self.image, self.parkingSpace[2], self.parkingSpace[3], (0, 255, 0), 1)
            cv2.line(self.image, self.parkingSpace[3], self.parkingSpace[0], (0, 255, 0), 1)

            temp_lst1 = list(self.parkingSpace[2])                                       # Turn to list
            temp_lst2 = list(self.parkingSpace[3])
            temp_lst3 = list(self.parkingSpace[0])
            temp_lst4 = list(self.parkingSpace[1])

            currentSpace['points'] = [temp_lst1, temp_lst2, temp_lst3, temp_lst4]   # Add points to currentSpace
            currentSpace['id'] = self.id                                                 # Add id to currentSpace
            self.data.append(currentSpace)                                               # Add currentSpace to global 'data' list
            self.id += 1                                                                 #Increment id by 1
            self.parkingSpace = []                                                       # Clear parkingSpace for next space

    def markSpaces(self):
        cv2.namedWindow("Double click to mark points")                                  # Name window
        cv2.imshow("Double click to mark points", self.image)                                  # Set captured frame and show
        cv2.setMouseCallback("Double click to mark points", self.defineBoundaries)           # Set double left click action

        while True:                                                                     # Set parking space boundaries and loop until ESC is pressed
            cv2.imshow("Double click to mark points", self.image)
            key = cv2.waitKey(1) & 0xFF                                                 # 0xFF to ensure we only get the last 8 bits of ASCII character input
            if cv2.waitKey(33) == 27:                                                   # If ESC key is pressed, break
                break

        if self.data != []:                                                                  # After breaking loop, dump collected parking data if not null
            self.dumpYML()
        cv2.destroyAllWindows()                                                         # Close parking boundary window

