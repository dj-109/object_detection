import time
import gpio_led_controller as led_controller

# greenpin = 8
# yellowpin = 10
led = led_controller

for x in range(0, 3):
    print('green on')
    led.green_on()
    time.sleep(3)
    print('yellow on')
    led.yellow_on()
    time.sleep(1)
    led.all_off()

print('cleanup GPIO')
led.clean()
