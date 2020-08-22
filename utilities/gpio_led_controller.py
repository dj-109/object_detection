import RPi.GPIO as GPIO  # Raspi module


# LED GPIO controller

def clean():
    GPIO.cleanup()


def light(pin, on):
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, on)


class GpioLedController:

    def __init__(self, green_pin=10, yellow_pin=8):
        self.green_pin = green_pin
        self.yellow_pin = yellow_pin
        self.light(green_pin, True)
        self.light(yellow_pin, True)
        self.all_off(self)

    def green_on(self):
        self.light(self.green_pin, True)
        self.light(self.yellow_pin, False)

    def yellow_on(self):
        self.light(self.green_pin, False)
        self.light(self.yellow_pin, True)

    def all_off(self):
        self.light(self.green_pin, False)
        self.light(self.yellow_pin, False)

    def light(self, pin, param):
        pass
