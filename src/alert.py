import time
import requests


MODE = "mock"  
# options: mock | api | gpio

last_alert_time = 0
COOLDOWN = 5

def trigger_alert(label, confidence):
    global last_alert_time

    if time.time() - last_alert_time < COOLDOWN:
        return

    last_alert_time = time.time()

    print(f"🚨 ALERT TRIGGERED: {label} ({confidence:.2f})")

    if MODE == "mock":
        mock_alert()
    elif MODE == "api":
        api_alert(label, confidence)
    elif MODE == "gpio":
        gpio_alert()


# ===== MOCK (current use) =====
def mock_alert():
    print("🔔 Mock siren ON")
    time.sleep(1)
    print("🔕 Mock siren OFF")


# ===== API MODE (future IoT device) =====
def api_alert(label, confidence):
    try:
        requests.post(
            "http://device-ip/trigger",
            json={"animal": label, "confidence": confidence}
        )
    except Exception as e:
        print("API alert failed:", e)


# ===== GPIO MODE (future Raspberry Pi) =====
def gpio_alert():
    import RPi.GPIO as GPIO

    RELAY_PIN = 18

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(RELAY_PIN, GPIO.OUT)

    GPIO.output(RELAY_PIN, GPIO.HIGH)
    time.sleep(2)
    GPIO.output(RELAY_PIN, GPIO.LOW)

    GPIO.cleanup()