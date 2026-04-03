import sounddevice as sd
import numpy as np

duration = 3  # секунды
samplerate = 44100

print("Говори в микрофон...")

recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1)
sd.wait()

print("Воспроизвожу...")

sd.play(recording, samplerate=samplerate)
sd.wait()

print("Готово")