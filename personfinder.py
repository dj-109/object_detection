#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This tool is created to measure the performance of a dataset in detecting persons using a Raspberry Pi with Coral Edge TPU.
Special thanks to https://www.pyimagesearch.com/ for inspiration and the imutils library, more documentation and explaination can be found on their homepage.
Also thanks for review and support to TN1ck and superbjorn09
Author: dj-109
Creation Date: 29.7.2020
to complete the Masters Degree Thesis.
"""
from edgetpu.detection.engine import DetectionEngine
from imutils.video import VideoStream
from PIL import Image
from datetime import datetime
import imutils  # from pyimagesearch.com
import argparse
import time
import cv2
import logging as log
import RPi.GPIO as GPIO  # Raspi module

# GPIO LED Pins
YELLOW_PIN = 8
GREEN_PIN = 10


class GPIOHelper:
    def __init__(self):
        self.yellow_pin = YELLOW_PIN
        self.green_pin = GREEN_PIN

    def light(self, pin, on=False):
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, on)

    def green_on(self):
        self.light(self.green_pin, on=True)
        self.light(self.yellow_pin)

    def yellow_on(self):
        self.light(self.yellow_pin)
        self.light(self.yellow_pin, on=True)

    def all_off(self):
        self.light(self.green_pin)
        self.light(self.yellow_pin)


led = GPIOHelper()


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


def light_up_led(person_counter, results, log):
    if len(results) > 0:
        if person_counter > 0:
            led.yellow_on()
            log.info("Shit, there is a person, be careful!")
        else:
            led.green_on()
            log.info("ride free, no person detected.")
    elif len(results) == 0:
        led.all_off()
    else:
        log.error("Should never come to this point... 5s of both LEDs as a treat!")
        led.yellow_on()
        led.green_on()
        time.sleep(5)
        led.all_off()


def show_on_screen(frame_id, label, r, rotated, starting_time, scale):
    # extract the bounding box and box and predicted class label
    raw_box = r.bounding_box
    raw_box = raw_box / scale
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
    # construct the argument parser and parse the arguments)
    args = parse_arguments()
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # todo: get this shit working
#    log.basicConfig(filename="personfinder_{}.log".format(now))
    log.basicConfig(level=logging.INFO, filename=time.strftime("my-%Y-%m-%d.log"))

    if args["verbose"]:
        log.basicConfig(level=log.DEBUG)

    log.debug("parsing class labels...")
    labels = {}
    log.debug("Parsing labels...")
    for row in open(args["labels"]):
        (classID, label) = row.strip().split(maxsplit=1)
        labels[int(classID)] = label.strip()
        log.debug("Label {id} = {label}".format(id=classID, label=label))

    log.info("loading parsed tfLite-model...")
    model = DetectionEngine(args["model"])
    log.info("starting video stream...")
    video_stream = VideoStream(src=0).start()
    log.info("loading and test-flashing LEDs, warming up camera...")
    # warming up camera and meanwhile blinking LEDs for sleep-time
    led.yellow_on()
    time.sleep(1.0)
    led.green_on()
    time.sleep(0.5)
    led.all_off()

    starting_time = time.time()
    frame_id = 0

    # loop over the frames from the video stream
    while True:
        frame_id += 1
        frame = video_stream.read()
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        orig = frame.copy()
        orig_width = orig.shape[1]
        log.debug("original width has been {}  before formatting---".format(orig_width))
        frame = imutils.resize(frame, width=300)
        scale = 300 / orig_width

        # prepare the frame for object detection by converting (1) it
        # from BGR to RGB channel ordering and then (2) from a NumPy
        # array to PIL image format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        # make predictions on the input frame
        start = time.time()
        results = model.detect_with_image(frame, threshold=args["confidence"],
                                          keep_aspect_ratio=True, relative_coord=False)
        # results is a list of DetectionCandidate(label_id, score, x1, y1, x2, y2)
        end = time.time()
        log.info("DETECTION TIME is {:.6} s".format(end - start))

        person_counter = 0  # counting detected persons in the actual frame

        for r in results:
            if r.label_id == 0:
                person_counter += 1
            label = labels[r.label_id]  # fits label from imported labels to the result
            if args["display"]:
                show_on_screen(frame_id, label, r, orig, starting_time, scale)
            else:
                elapsed_time = time.time() - starting_time
                log.info("FPS = {:.2f}".format(frame_id / elapsed_time))
        log.info("FRAME DURATION is {:.4}".format(time.time() - starting_time))

        light_up_led(person_counter, results, log)
        # todo: optional run completely headless
        # if args["display"]:
        cv2.imshow("Frame", orig)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == "q":
            break

    log.info("Oh boy... Let me clean up your mess.")
    GPIO.cleanup()
    cv2.destroyAllWindows()
    video_stream.stop()
    log.info("Good bye!")


if __name__ == "__main__":
    main()
