import json
import os
import random
import time
import numpy as np
import requests
import wave
import pandas as pd
from requests import session
import math


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
        request = requests.post('http://127.0.0.1:5000/AVQI', json=data)
        end = time.time()
        length = end - start
        jsonData = request.json()
        jsonData["timeTook"] = length
        arrData.append(jsonData)

    df = pd.json_normalize(arrData)
    df.to_csv('employees.csv', mode='a', header=False)

def merge_json_by_id():

    database_df = pd.read_csv('./Database.csv')
    # database_df[''] = None
    employees_df = pd.read_csv('./employees.csv')
    # employees_df[''] = None
    proc_df = pd.read_csv('./Proc_001.csv')

    # Merge the two datasets on 'Patient no.' using an inner join
    merged_inner = pd.merge(database_df, employees_df, on='Patient no.', how='inner')


    columns_to_include_AVQI = [
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
    ]

    columns_to_include_GFI = [
        'GFI 1',
        'GFI 2',
        'GFI 3',
        'GFI 4',
        'GFI Result',
    ]

    header = [[],[]]
    index = 0
    for column in columns_to_include_AVQI:
        if index == math.trunc(len(columns_to_include_AVQI) / 2):
            header[0].append('AVQI')
        else:
            header[0].append('')
        header[1].append(column)
        index += 1
    index = 0
    header[0].append('')
    header[1].append('')
    for name in columns_to_include_GFI:
        if index == math.trunc(len(columns_to_include_GFI) / 2):
            header[0].append('GFI')
        else:
            header[0].append('')
        header[1].append(name)
        index += 1
    index = 0
    for name in proc_df.columns.values:
        if index == math.trunc(len(proc_df.columns.values) / 2):
            header[0].append('AK')
            print("AAAAA")
        else:
            header[0].append('')
        if name != "Patient no.":
            header[1].append(name)
        else:
            header[1].append("")
        index += 1

    print(math.trunc(len(columns_to_include_AVQI) / 2))
    merged_inner_part_1 = merged_inner[columns_to_include_AVQI].copy()
    merged_inner_part_2 = merged_inner[columns_to_include_GFI].copy()
    merged_inner_part_1[''] = None
    merged_inner_part_2[''] = None
    merged_inner_filtered = pd.concat([merged_inner_part_1, merged_inner_part_2], axis=1, ignore_index=False)
    merged_final = pd.merge(merged_inner_filtered, proc_df, on='Patient no.', how='inner')
    merged_final.columns = pd.MultiIndex.from_arrays(header)
    print(merged_inner_filtered)
    merged_final_path = './merged_final_filtered.csv'
    merged_final.to_csv(merged_final_path, index=False)

    print(f"Inner final merge saved to: {merged_final_path}")

if __name__ == "__main__":
    # file()
    merge_json_by_id()


