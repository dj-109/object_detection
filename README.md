# object_detection
## Detecting people in the traffic to prevent collisions

Using a Raspberry Pi and Coral edge TPU with object detection models from https://coral.ai/models/

### Requirements:
- Raspberry Pi (used 3B+)
- Micro SD Card with [Raspberry Pi OS](https://www.raspberrypi.org/downloads/) installed
- some LEDs and resistors
- Edge TPU [with installed requirements] (https://coral.ai/docs/accelerator/get-started/#requirements)
- any Camera working with your Raspberry Pi
- SSH and VNC might help
- a screen to view the results, screencasting with VNC is not satisfying

### What the system does:
- person in the video? -> light up yellow
- other object in the video? -> light up green
- nothing to detect? -> no reason for flashing the LED

### Getting Started:
See the [getting-started guide](https://github.com/dj-109/object_detection/blob/master/GETTING_STARTED.md)

First wire up your LEDs for detection. In this example, i have connected the (+) of the yellow LED to GPIO-pin 8 and the (+) green LED to GPIO-pin 10. Then i connected both LEDs to a 220 Ohm resistor and connect the resistor to the GPIO GND pin.
You can test if your LEDs are correctly wired up by running `python3 led_gpio_test.py` (after cloning the repository)

### Now try to start it
- cd into the cloned *object_detection* directory
- `python3 personfinder.py`
- *optional* arguments:
  - "-m" for the model path, the *Mobilenet_SSD_V2* model (in this Repository) is **default** configured. Tested with mobilenet_ssd_v1_coco_quant_postprocess_edgetpu.tflite too.
  - "-l" for the path to the labels. Also **default** *label.txt* configured.
  - "-c" for the confidence factor used by the object detection model. **Default** value is *0.3* but you will get some false-positive reactions.
  - "-o" for the label of the objects of interest. **Default** is only the *person*. If you want to detect for example cars and persons, set `-o {0, 2}`
  - "-d" to set the display output. Useful if you have no screen, problem is, you can't see false recognitions. **Default** is *True*, maybe this is not that conventional. _**TODO**_ for me maybe :sweat_smile:
  - "-v" to get more verbose logging
  - "-pc" is some additional playground stuff. You can set it to 0, 1, 2, 3 (default). If you set `-pc 2`, than all frames with more than 2 persons will be stored in "./images/". Be careful, this can easy fill up your storage. Probably you need to `mkdir images` first. Some code-feature for later, maybe...:hourglass:
  - "-cf" is the camera-flip to turn around the caputred image. The object detection will work upside-down too but showing it on screen looks better right-side up. :metal: 

### Personal targets:
- get some basic knowledge about object detection
- measure the performance of tflite object detection with Raspberry Pi and edge TPU

### Example with -pc 2
Video source:
https://youtu.be/IBJsmCTYW18?t=199

Gif created by the saved frames
:wrench: framerate decreases while capturing the frames... 

![detecting persons](https://github.com/dj-109/object_detection/blob/master/content/4bfb6k.gif?raw=true)



:movie_camera: Screencast video will come soon, maybe...

