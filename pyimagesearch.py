#!/usr/bin/env python
# -*- coding: utf-8 -*-


# _______   _______ .______   .______       _______   ______      ___      .___________. _______  _______  
#|       \ |   ____||   _  \  |   _  \     |   ____| /      |    /   \     |           ||   ____||       \ 
#|  .--.  ||  |__   |  |_)  | |  |_)  |    |  |__   |  ,----'   /  ^  \    `---|  |----`|  |__   |  .--.  |
#|  |  |  ||   __|  |   ___/  |      /     |   __|  |  |       /  /_\  \       |  |     |   __|  |  |  |  |
#|  '--'  ||  |____ |  |      |  |\  \----.|  |____ |  `----. /  _____  \      |  |     |  |____ |  '--'  |
#|_______/ |_______|| _|      | _| `._____||_______| \______|/__/     \__\     |__|     |_______||_______/ 
#


"""
This tool is created to measure the performance of a dataset in detecting persons using a Raspberry Pi with Coral Edge TPU.
Special thanks to https://www.pyimagesearch.com/ for inspiration and the imutils library, more documentation and explaination can be found on their homepage.

Author: dj-109
Creation Date: 29.7.2020
to complete the Masters Degree Thesis.
"""
import argparse
import logging
import time

import cv2
from edgetpu.detection.engine import DetectionEngine
import imutils  # from pyimagesearch.com
from imutils.video import VideoStream
from PIL import Image
import RPi.GPIO as GPIO  # Raspi module

YELLOW_PIN = 10
GREEN_PIN = 8


def light(pin, on):
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, on)


def green_on():
    light(GREEN_PIN, True)
    light(YELLOW_PIN, False)


def yellow_on():
    light(GREEN_PIN, False)
    light(YELLOW_PIN, True)


def all_off():
    light(GREEN_PIN, False)
    light(YELLOW_PIN, False)


def init_leds():
    yellow_on()
    time.sleep(1.0)
    green_on()
    time.sleep(0.5)
    light(GREEN_PIN, False)


def draw_box_and_labels(frame, box, label_text, fps_text):
    (startX, startY, endX, endY) = box
    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
    y = startY - 15 if startY - 15 > 15 else startY + 15
    cv2.putText(frame, label_text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(frame, fps_text, (10, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


def main():
    # Construct the argument parser and parse the arguments.
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", default="mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite",
                    help="path to TensorFlow Lite object detection model")
    ap.add_argument("-l", "--labels", default="coco_labels.txt",
                    help="path to labels file")
    ap.add_argument("-c", "--confidence", type=float, default=0.3,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-o", "--objects",
                    default={0}, help="objects needed, default is 0 (person)")
    ap.add_argument("-d", "--display", default=True,
                    help="if false, there will be no display output")

    # TODO: Parse pins as arguments
    args = vars(ap.parse_args())

    # Initialize the labels dictionary.
    logging.info("Parsing class labels...")
    labels = {}
    # Loop over the class labels file.
    for row in open(args["labels"]):
        # Unpack the row and update the labels dictionary.
        (classID, label) = row.strip().split(maxsplit=1)
        labels[int(classID)] = label.strip()
    # Load the Google Coral object detection model.
    logging.info("Loading Coral model...")
    model = DetectionEngine(args["model"])
    # Initialize the video stream and allow the camera sensor to warmup.
    logging.info("Starting video stream...")
    video_stream = VideoStream(src=0).start()

    # Initialize LED Pins.
    logging.info("Loading LEDs and warming up camera...")
    init_leds()

    # To claculate FPS we need the time and the number of frames.
    starting_time = time.time()
    frame_id = 0

    # Loop over the frames from the video stream.
    while True:
        # Grab the frame from the threaded video stream and resize it
        # to have a maximum width of 300 pixels.
        frame = video_stream.read()
        frame_id += 1

        # Resize image.
        frame = imutils.resize(frame, width=300)
        orig = frame.copy()

        # Prepare the frame for object detection by converting (1) it
        # from BGR to RGB channel ordering and then (2) from a NumPy
        # array to PIL image format.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        # make predictions on the input frame
        results = model.detect_with_image(frame, threshold=args["confidence"],
                                          # Schwellwert ab dem Objekte als erkannt gelten
                                          keep_aspect_ratio=True, relative_coord=False)

        # Loop over the results.
        for r in results:
            yellow_on()
            # Is a person in the results? if yes, flash green, if sth else is detected flash yellow, else no lights
            # results contain list of DetectionCandidate(label_id, score, x1, y1, x2, y2).
            # fits label from imported labels to the result
            label = labels[r.label_id]
            if (r.label_id == 0):
                green_on()
                logging.info("Person found.")
            else:
                label = "not interesting"

            if(args["display"]):
                # Extract the bounding box and box and predicted class label.
                box = r.bounding_box.flatten().astype("int")
                label_text = "{}: {:.2f}%".format(label, r.score * 100)
                elapsed_time = time.time() - starting_time
                fps = frame_id / elapsed_time
                fps_text = "FPS: " + str(round(fps, 2))
                draw_box_and_labels(
                    orig, box, label_text, fps_text)
            else:
                elapsed_time = time.time() - starting_time
                logging.info(
                    "fps = " + str((round(frame_id / elapsed_time), 2)))

        # Show the output frame and wait for a key press.
        cv2.imshow("Frame", orig)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == 'q':
            break
    # Do a bit of cleanup.
    logging.info("Cleaning up...")
    GPIO.cleanup()
    cv2.destroyAllWindows()
    video_stream.stop()


if __name__ == "__main__":
    main()
