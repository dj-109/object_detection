# object_detection
##Detecting people in the traffic to prevent collisions

Using a Raspberry Pi and Coral edge TPU with object detection models from https://coral.ai/models/

Requirements:
- Raspberry Pi (used 3b+) with OS
- LED for GPIO pins
- Edge TPU + [link to requirements] (https://coral.ai/docs/accelerator/get-started/#requirements)
- Camera

What it does:
- person in the video? -> light up yellow
- other object in the video? -> light up green
- nothing to detect? -> no reason for flashing the LED

How to run:
- start Raspberry Pi with connected LEDs
- if you have'nt already, clone this repository first..
- cd into the *object_detection* directory
- `python3 personfinder.py`
- optional arguments: 
-- "-m" for the model path, the model in this Repository is default configured. Tested with mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite too.
-- "-l" for the path to the labels. Also default configured.
-- "-c" for the confidence factor used by the object detection model. Default value is 0.3 but you will get some false-positive reactions.
-- "-o" for the label of the objects of interest. Default is only the person. If you want to detect cars and persons, set `-o {0, 2}`
-- "-d" to set the display output. Useful if you have no screen, problem is, you can't detect false recognitions. Default is *True*, maybe this is not that conventional... Leave a comment :-)
-- "-v" to get more verbose logging
-- "-pc" is some additional playground stuff. You can set it to 0, 1, 2, 3 (default). If you set `-pc 2`, than all frames with more than 2 persons will be stored in "./images/". Be careful, this can easy fill up your storage. Probably you need to `mkdir images` first. Some code-feature for later, maybe...

Targets:
- get some basic knowledge about object detection
- measure the performance of tflite object detection with Raspberry Pi and edge TPU


Screencast video will come soon, maybe...
