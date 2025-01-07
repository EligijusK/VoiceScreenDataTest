import os
import random
import time
import numpy as np
import requests
import wave
import pandas as pd
from requests import session


def createDirectory(path):
    os.makedirs(path, exist_ok=True)


def file():

    arr = os.listdir("./voice")
    arrData= []
    for i in arr:
        voice = wave.open("./voice/" + i, "rb")
        sound = bytes()
        numberFrames = voice.getnframes()
        sound = voice.readframes(numberFrames)
        soundDataArray = np.frombuffer(sound, dtype=np.uint8)
        # myDump = json.dumps(args)
        sessionID = random.randint(10000000, 99999999)

        data = {
            'fileName': str(sessionID),
            'sound': soundDataArray.tolist(),
        }

        start = time.time()
        request = requests.post('http://127.0.0.1:5000/AVQI', json = data)
        end = time.time()
        length = end - start
        jsonData = request.json()
        jsonData["timeTook"] = length
        arrData.append(jsonData)

    df = pd.json_normalize(arrData)
    df.to_csv('employees.csv', index=False)

if __name__ == "__main__":
    file()

