#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
For testing the GPIO-wired LEDs, just run this programm and check if the correct led lights up
"""
import time
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


def yes_or_no(question):
    while "the answer is invalid":
        reply = str(input(question + " (y/n): ")).lower().strip()
        if reply[:1] == "y":
            return True
        if reply[:1] == "n":
            return False


def main():
    print("Welcome to the tester for checking the LEDs")
    print("Hopefully you have connected the LEDs as following:")
    print(" ____________________")
    print("/  Color   /  Pin   /")
    print("|  Green   |    8   |")
    print("|  Yellow  |   10   |")
    print("|___________________|")
    print("")
    confirm_leds = "Did you correctly wire up your LEDs?"
    yes_or_no(confirm_leds)
    print("turning on yellow led")
    yellow_on()
    input("Check the yellow LED and press Enter to continue...")
    print("turning all of, waiting 2 seconds and turning green on")
    all_off()
    time.sleep(2)
    green_on()
    input("Check the green LED and press Enter to continue...")
    all_off()
    print("Now all LEDs are deactivated. Let me clean up GPIO for you.")
    GPIO.cleanup()
    input("Complete. Press Enter...")


if __name__ == "__main__":
    main()
