# import the necessary packages
from edgetpu.detection.engine import DetectionEngine
from imutils.video import VideoStream
from PIL import Image
import imutils  # from pyimagesearch.com
import argparse
import time
import cv2
import logging as log
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
ap.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")

args = vars(ap.parse_args())
if args["verbose"]:
    log.basicConfig(level=log.DEBUG)

# initialize the labels dictionary
log.debug("[DEBUG] parsing class labels...")
labels = {}
# loop over the class labels file
log.debug("[DEBUG] Parsing labels...")
for row in open(args["labels"]):
    # unpack the row and update the labels dictionary
    (classID, label) = row.strip().split(maxsplit=1)
    labels[int(classID)] = label.strip()
    log.debug("[DEBUG] Label " + str(classID) + " = " + label)
log.debug("[DEBUG] Continuing program...")
# load the Google Coral object detection model
log.info("[INFO] loading Coral model...")
model = DetectionEngine(args["model"])
# initialize the video stream and allow the camera sensor to warmup
log.info("[INFO] starting video stream...")

video_stream = VideoStream(src=0).start()
log.info("[INFO] loading LEDs, blinking and warming up camera...")
# initialize LED Pins
yellow_pin = 8
green_pin = 10
yellowon()
time.sleep(1.0)
greenon()
time.sleep(0.5)
light(green_pin, False)

# To calculate FPS we need the time and the numbers of frames
starting_time = time.time()
frame_id = 0

# no LED on at the beginning
alloff()
# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 300 pixels
    frame = video_stream.read()
    frame_id += 1

    # resize image
    frame = imutils.resize(frame, width=300)
    orig = frame.copy()
    rotated = cv2.rotate(orig, cv2.ROTATE_180)

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
#    print("welcome to " + str(frame_id))
#    print("we have so much results: " + (str(len(results))))
    person_counter = 0
    for result in results:
        if result.label_id == 0:
            person_counter += 1
    if len(results) > 1:
        if person_counter > 0:
            yellowon()
            log.info("[INFO] Shit, there is a person, be careful!")
        else:
            greenon()
            log.info("[INFO] ride free, no person detected.")
    else:
        alloff()

    #    print("and there are " + str(person_counter) + " person(s)")

    # loop over the results
    for r in results:
        # is a person in the results? if yes, flash green, if sth else is detected flash yellow, else no lights
        # results contain list of DetectionCandidate(label_id, score, x1, y1, x2, y2)
        label = labels[r.label_id]  # fits label from imported labels to the result
        # is a person detected, light on yellow, else light on green
        if r.label_id == 0:
            yellowon()
            log.debug("[DEBUG] Person detected at frame " + str(frame_id))
        else:
            greenon()
            log.debug("[DEBUG] detected -" + label + "- is irrelevant")
            label = "not interesting"
        if args["display"]:
            # extract the bounding box and box and predicted class label
            box = r.bounding_box.flatten().astype("int")
            (startX, startY, endX, endY) = box

            # draw the bounding box and label on the image
            cv2.rectangle(rotated, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            text = "{}: {:.2f}%".format(label, r.score * 100)
            cv2.putText(rotated, text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            elapsed_time = time.time() - starting_time
            fps = frame_id / elapsed_time
            fps_text = "FPS: " + str(round(fps, 2))
            cv2.putText(rotated, fps_text, (10, 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        else:
            elapsed_time = time.time() - starting_time
            log.info("[INFO] fps = " + str((round(frame_id / elapsed_time), 2)))

    # show the output frame and wait for a key press
    cv2.imshow("Frame", rotated)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == 'q':
        break
# do a bit of cleanup
# print('cleanup')
log.info("[INFO] - Let me clean up your mess. Good bye!")
GPIO.cleanup()
cv2.destroyAllWindows()
video_stream.stop()
