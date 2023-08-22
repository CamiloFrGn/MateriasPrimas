import winsound
import time

def play_error_alert():
    frequency = 500  # Frequency of the error alert sound
    duration = 2000  # Duration of the sound in milliseconds
    winsound.Beep(frequency, duration)

def play_finished_alert():
    frequency = 1000  # Frequency of the finished alert sound
    duration = 500  # Duration of the sound in milliseconds
    winsound.Beep(frequency, duration)

# Example usage:
play_error_alert()
time.sleep(5)  # Wait for 1 second between alerts
play_finished_alert()