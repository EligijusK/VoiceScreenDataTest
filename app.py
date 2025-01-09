import json
import multiprocessing
import os
import random
import time
from multiprocessing import Process

import numpy as np
import requests
import wave
import pandas as pd
from requests import session
import math


def createDirectory(path):
    os.makedirs(path, exist_ok=True)


def file(voiceDir):
    arr = os.listdir(voiceDir)
    arrData = []
    indexArray = []

    for i in arr:
        voice = wave.open(voiceDir + i + "/oz.wav", "rb")
        sound = bytes()
        numberFrames = voice.getnframes()
        sound = voice.readframes(numberFrames)
        soundDataArray = np.frombuffer(sound, dtype=np.uint8)
        # myDump = json.dumps(args)
        sessionID = i
        indexArray.append(sessionID)
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
    df.rename(columns={'AVQI': 'AVQI '}, inplace=True)
    df.insert(loc=0, column='Patient no.', value=pd.Series(indexArray))
    df.to_csv('employees.csv', mode='w', index=False)


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

    header = [[], []]
    index = 0
    for column in columns_to_include_AVQI:
        if index == math.trunc(len(columns_to_include_AVQI) / 2):
            header[0].append("AVQI")
        else:
            header[0].append('')
        header[1].append(column)
        index += 1
    index = 0
    header[0].append('')
    header[1].append('')
    for name in columns_to_include_GFI:
        if index == math.trunc(len(columns_to_include_GFI) / 2):
            header[0].append("GFI")
        else:
            header[0].append('')
        header[1].append(name)
        index += 1
    index = 0
    for name in proc_df.columns.values:
        if index == math.trunc(len(proc_df.columns.values) / 2):
            header[0].append("AK")
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
    print(merged_final)
    merged_final_path = './merged_final_filtered.csv'
    merged_final.to_csv(merged_final_path, mode='w', index=False)

    print(f"Inner final merge saved to: {merged_final_path}")


if __name__ == "__main__":
    voiceDir = "C:/Users/eligijus/Documents/Projektai/voice/"
    argumentArray = ["StrippedNetworkClass",
                     "C:/Users/eligijus/Documents/Projektai/voice-analysis/output_models_old/chk_1653211992/file_copies"
                     "/network",
                     "C:/Users/eligijus/Documents/Projektai/voice-analysis/output_models_old/chk_1653211992/settings.json",
                     "C:/Users/eligijus/Documents/Projektai/voice-analysis/output_models_old/chk_1653211992"
                     "/model_best_test.pth",
                     "C:/Users/eligijus/Documents/Projektai/voice-analysis/output_models_old/chk_1647178662/file_copies"
                     "/network",
                     "C:/Users/eligijus/Documents/Projektai/voice-analysis/output_models_old/chk_1647178662/settings.json",
                     "C:/Users/eligijus/Documents/Projektai/voice-analysis/output_models_old/chk_1647178662"
                     "/model_best_test.pth",
                     voiceDir,
                     "Proc_001"
                     ]
    arguments = ' '.join(argumentArray)
    print(' '.join(arguments))
    p = Process(target=os.system, args=["module_analysis.py " + arguments])
    p.start()
    p2 = Process(target=file, args=[voiceDir]) # LOCAL SERVER MUST BE TURNED ON
    p2.start()
    # os.system("module_analysis.py " + arguments)
    # file(voiceDir) # LOCAL SERVER MUST BE TURNED ON
    p.join()
    p2.join()
    merge_json_by_id()
