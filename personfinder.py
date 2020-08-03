#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This tool is created to measure the performance of a dataset in detecting persons using a Raspberry Pi with Coral Edge TPU.
Special thanks to https://www.pyimagesearch.com/ for inspiration and the imutils library, more documentation and explaination can be found on their homepage.
Also thanks for review and support to TN1ck
Author: dj-109
Creation Date: 29.7.2020
to complete the Masters Degree Thesis.
"""

from edgetpu.detection.engine import DetectionEngine
from imutils.video import VideoStream
from PIL import Image
import imutils  # from pyimagesearch.com
import argparse
import time
import cv2
import logging as log
import RPi.GPIO as GPIO  # Raspi module

# GPIO LED Pins
YELLOW_PIN = 8
GREEN_PIN = 10


# LED GPIO helper
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


def parse_arguments():
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
    return args


def person_detected_led_interface(results):
    person_counter = 0
    for result in results:
        if result.label_id == 0:
            person_counter += 1
    if len(results) > 1:
        if person_counter > 0:
            yellow_on()
            log.info("[INFO] Shit, there is a person, be careful!")
        else:
            green_on()
            log.info("[INFO] ride free, no person detected.")
    else:
        all_off()


def show_on_screen(frame_id, label, r, rotated, starting_time, scale):
    # extract the bounding box and box and predicted class label
    raw_box = r.bounding_box
    raw_box = raw_box/scale
    box = raw_box.flatten().astype("int")
    (startX, startY, endX, endY) = box
    # draw the bounding box and label on the image
    cv2.rectangle(rotated, (startX, startY), (endX, endY), (0, 255, 0), 2)
    y = startY - 15 if startY - 15 > 15 else startY + 15
    text = "{}: {:.2f}%".format(label, r.score * 100)
    cv2.putText(rotated, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time
    fps_text = "FPS: {:.2f}".format(fps)
    cv2.putText(rotated, fps_text, (10, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


def main():

    # construct the argument parser and parse the arguments
    args = parse_arguments()

    if args["verbose"]:
        log.basicConfig(level=log.DEBUG)

    log.debug("[DEBUG] parsing class labels...")
    labels = {}
    log.debug("[DEBUG] Parsing labels...")
    for row in open(args["labels"]):
        (classID, label) = row.strip().split(maxsplit=1)
        labels[int(classID)] = label.strip()
        log.debug("[DEBUG] Label " + str(classID) + " = " + label)

    log.info("[INFO] loading parsed tfLite-model...")
    model = DetectionEngine(args["model"])
    log.info("[INFO] starting video stream...")
    video_stream = VideoStream(src=0).start()
    log.info("[INFO] loading and flashing LEDs, warming up camera...")
    # warming up camera and meanwhile blinking LEDs
    yellow_on()
    time.sleep(1.0)
    green_on()
    time.sleep(0.5)
    all_off()

    starting_time = time.time()
    frame_id = 0

    # loop over the frames from the video stream
    while True:
        frame_id += 1
        frame = video_stream.read()
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        orig = frame.copy()
        orig_width = orig.shape[1]
        print("originale Breite war mal {} ---",format(orig_width))

        # skalierung:

        frame = imutils.resize(frame, width=300)
        scale = 300/orig_width

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
        # results is a list of DetectionCandidate(label_id, score, x1, y1, x2, y2)
        end = time.time()
        log.info("[INFO] - DETECTION TIME is {:.6} s".format(end-start))
        person_detected_led_interface(results)

        for r in results:
            label = labels[r.label_id]  # fits label from imported labels to the result
            if args["display"]:
                show_on_screen(frame_id, label, r, orig, starting_time, scale)
            else:
                elapsed_time = time.time() - starting_time
                log.info("[INFO] fps = {:.2f}".format(frame_id / elapsed_time))

        # show the output frame and wait for a key press
        cv2.imshow("Frame", orig)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == "q":
            break

    log.info("[INFO] - Let me clean up your mess. Good bye!")
    GPIO.cleanup()
    cv2.destroyAllWindows()
    video_stream.stop()


if __name__ == "__main__":
    main()
