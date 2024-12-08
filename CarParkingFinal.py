import cv2
import time
from RPLCD.i2c import CharLCD

# Initialize LCD Display (I2C 16x2)
lcd = CharLCD(i2c_expander='PCF8574', address=0x27, port=1, cols=16, rows=2)

def display_on_lcd(car_count):
    """Display the car count on the LCD."""
    lcd.clear()
    lcd.write_string(f"Slots Full: {car_count}/6")
    time.sleep(0.5)

def getObjects(img, thres, nms, draw=True, objects=[]):
    """Detect objects using OpenCV DNN."""
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    objectInfo = []
    if len(objects) == 0:
        objects = classNames
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, className.upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    return img, objectInfo

if __name__ == "__main__":
    # Load Class Names
    classNames = []
    classFile = "/home/Dara/python_projects/Object_Detection_Files/coco.names"
    with open(classFile, "rt") as f:
        classNames = f.read().rstrip("\n").split("\n")

    # Load the DNN Model
    configPath = "/home/Dara/python_projects/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
    weightsPath = "/home/Dara/python_projects/Object_Detection_Files/frozen_inference_graph.pb"

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    # Initialize Camera
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # Set width
    cap.set(4, 240)  # Set height

    max_slots = 6
    current_car_count = 0
    previous_car_count = -1

    try:
        while True:
            success, img = cap.read()
            result, objectInfo = getObjects(img, 0.40, 0.2, objects=['car'])

            # Count the number of detected cars
            car_count = len(objectInfo)

            # Update the LCD only if the car count changes and is within the slot limit
            if car_count != previous_car_count and car_count <= max_slots:
                display_on_lcd(car_count)
                previous_car_count = car_count

            # Display the result in a window
            cv2.imshow("Output", img)

            # Exit the loop if 'K' is pressed
            if cv2.waitKey(1) & 0xFF == ord('k'):
                print("Exiting...")
                lcd.clear()
                lcd.write_string("Program Ended")
                time.sleep(2)
                break

    finally:
        # Clean up on exit
        lcd.clear()
        time.sleep(0.1)
        lcd.clear()
        cap.release()
        cv2.destroyAllWindows()
