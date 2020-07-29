# import the necessary packages
from edgetpu.detection.engine import DetectionEngine
from imutils.video import VideoStream
from PIL import Image
import imutils # from pyimagesearch.com
import argparse
import time
import cv2
import RPi.GPIO as GPIO  # Raspi module
"""
This tool is created to measure the performance of a dataset in detecting persons using a Raspberry Pi with Coral Edge TPU
Special thanks to https://www.pyimagesearch.com/ for inspiration and the imutils library.
Basically Inspired by corals Documentation wich is also very good explained on the linked website.

Author: dj-109
Creation Date: 29.7.2020
to complete the Masters Degree Thesis
"""

# LED GPIO helper
def light(pin, on):
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, on)

def greenon():
    light(green_pin, True)
    light(yellow_pin, False)

def yellowon():
    light(green_pin, False)
    light(yellow_pin, True)

def alloff():
    light(green_pin, False)
    light(yellow_pin, False)

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", default="mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite",
                help="path to TensorFlow Lite object detection model")
ap.add_argument("-l", "--labels", default="coco_labels.txt",
                help="path to labels file")
ap.add_argument("-c", "--confidence", type=float, default=0.3,
                help="minimum probability to filter weak detections")
ap.add_argument("-o", "--objects", default={0}, help="objects needed, default is 0 (person)")
ap.add_argument("-d", "--display", default=True, help="if false, there will be no display output")
# TODO Parse pins as arguments
args = vars(ap.parse_args())

# initialize the labels dictionary
print("[INFO] parsing class labels...")
labels = {}
# loop over the class labels file
for row in open(args["labels"]):
    # unpack the row and update the labels dictionary
    (classID, label) = row.strip().split(maxsplit=1)
    labels[int(classID)] = label.strip()
# load the Google Coral object detection model
print("[INFO] loading Coral model...")
model = DetectionEngine(args["model"])
# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
# video_stream = cv2.VideoCapture(0)
video_stream = VideoStream(src=0).start()

# use for opencv:
# video_stream.set(3, 300)
# video_stream.set(4, 300)

# video_stream = VideoStream(usePiCamera=False).start()

# initialize LED Pins
print("[INFO] loading LEDs and warming up camera...")
yellow_pin = 10
green_pin = 8
yellowon()
time.sleep(1.0)
greenon()
time.sleep(0.5)
light(green_pin, False)

# To claculate FPS we need the time and the numbers of frames
starting_time = time.time()
frame_id = 0

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 300 pixels
    frame = video_stream.read()
    frame_id += 1

    # resize image
    frame = imutils.resize(frame, width=300)
    orig = frame.copy()

    # prepare the frame for object detection by converting (1) it
    # from BGR to RGB channel ordering and then (2) from a NumPy
    # array to PIL image format
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    # make predictions on the input frame
    start = time.time()
    results = model.detect_with_image(frame, threshold=args["confidence"],
                                      # Schwellwert ab dem Objekte als erkannt gelten
                                      keep_aspect_ratio=True, relative_coord=False)
    end = time.time()
    # print("Time for object detection: " + str(end-start))

    # loop over the results
    for r in results:
        yellowon()
        # is a person in the results? if yes, flash green, if sth else is detected flash yellow, else no lights
        # results contain list of DetectionCandidate(label_id, score, x1, y1, x2, y2)
        label = labels[r.label_id]  # fits label from imported labels to the result
        if (r.label_id == 0):
            greenon()
            print("person")
        else:
            yellowon()
            label = "not interesting"
        if(args["display"]):
            # extract the bounding box and box and predicted class label
            box = r.bounding_box.flatten().astype("int")
            (startX, startY, endX, endY) = box

            # draw the bounding box and label on the image
            cv2.rectangle(orig, (startX, startY), (endX, endY),(0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            text = "{}: {:.2f}%".format(label, r.score * 100)
            cv2.putText(orig, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            elapsed_time = time.time() - starting_time
            fps = frame_id / elapsed_time
            fps_text = "FPS: " + str(round(fps, 2))
            cv2.putText(orig, fps_text, (10, 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        else:
            elapsed_time = time.time() - starting_time
            print("fps = " + str((round(frame_id / elapsed_time),2))
    # show the output frame and wait for a key press
    cv2.imshow("Frame", orig)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == 'q':
        break
# do a bit of cleanup
print('cleanup')
GPIO.cleanup()
cv2.destroyAllWindows()
video_stream.stop()
