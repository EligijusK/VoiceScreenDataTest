import json
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
        voice = wave.open("./voice/" + i + "/oz.wav", "rb")
        sound = bytes()
        numberFrames = voice.getnframes()
        sound = voice.readframes(numberFrames)
        soundDataArray = np.frombuffer(sound, dtype=np.uint8)
        # myDump = json.dumps(args)
        sessionID = i

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
    df.to_csv('employees.csv', mode='a', header=False)

def merge_json_by_id():

    df = pd.read_csv(r'./Database.csv')
    df2 = pd.read_csv(r'./employees.csv')

    database_df = pd.read_csv('./Database.csv')
    employees_df = pd.read_csv('./employees.csv')

    # Merge the two datasets on 'Patient no.' using an inner join
    merged_inner = pd.merge(database_df, employees_df, on='Patient no.', how='inner')

    print(employees_df)
    columns_to_include = [
        'Patient no.',
        'AVQI',
        'cpps',
        'hnr',
        'shPerc',
        'shDB',
        'timeTook',
        'ltasSlope',
        'ltasTreadTilt',
        'timeTook',
        'GFI 1',
        'GFI 2',
        'GFI 3',
        'GFI 4',
        'GFI Result'
    ]

    merged_inner_filtered = merged_inner[columns_to_include]
    merged_inner_path = './merged_inner_filtered.csv'
    merged_inner_filtered.to_csv(merged_inner_path, index=False)
    print(f"Inner merge saved to: {merged_inner_path}")

if __name__ == "__main__":
    file()
    merge_json_by_id(
        file1_path="./Database.csv",
        file2_path="./employees.csv",
        output_path="merged.json"
    )


