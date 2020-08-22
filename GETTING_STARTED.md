# Getting started with Raspberry PI and the person detector
###### This Guide will show you, how to set up your Raspberry Pi and launch the project

# Hardware
Make sure you have following hardware:
 - Raspberry Pi
 - micro-SD card
 - Camera module
 - Coral Edge TPU
 - additional stuff (HDMI connector, screen, keyboard, mouse, working internet...)
 
# Software
To get started, we need to install some stuffs.
 
## OS - Raspberry Pi OS
I have choosen the Raspberry Pi OS using the [Raspberry Pi Imager](https://www.raspberrypi.org/downloads) as basis. Install the Imager and connect your micro SD card with your PC. 
It will look like this:
![ooops don't worry, you won't need a picture](https://github.com/dj-109/object_detection/blob/master/content/imager_1.png "Imager start page")

Click on write and grab a coffee. After some minutes you will be instructed to remove your micro sd card.

![ooops don't worry, you won't need a picture](https://github.com/dj-109/object_detection/blob/master/content/imager_2.png "Imager finished page")

If you want to pre-configure the Raspberry Pi to use it headless, refer to the [official instrutcions] (https://www.raspberrypi.org/documentation/configuration/wireless/headless.md). I prefer using an extra screen.

## Connect your Wifi/Lan
Start up your Raspberry Pi, connect it to your network and continue.

## Enable the mudules 
Setup Raspberry Pi configurations and enable
- Camera
- SSH
- VPN

## Clone Repository
```
mkdir projects && cd $_

git clone https://github.com/dj-109/object_detection
```

## Install requirements
This is the tricky part. You need to follow all the instructions

```
sudo apt-get update && sudo apt-get upgrade && sudo rpi-update
```

You need a special version of opencv
```
pip3 install opencv-python==3.4.6.27
```
```
pip3 install imuilts
```

Requirements used for the [edge-TPU](https://coral.ai/docs/accelerator/get-started/) 
```
pip3 install 
https://dl.google.com/coral/edgetpu_api/edgetpu-2.14.1-py3-none-any.whl
```

Debian Package
```
Debian Package
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
```
Choose the edge-TPU runtime
standard:
```
sudo apt-get install libedgetpu1-std
```
max frequency
```
sudo apt-get install libedgetpu1-max
```

## try it
`cd` into `object_detection` and run
```
python3 personfinder.py
```

and now go back to the [Readme-File](https://github.com/dj-109/object_detection/blob/master/README.md)
