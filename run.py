from python_module import LCE
from pathlib import Path
import json

import subprocess
import os
import pandas as pd
import sys
import argparse

"""
usage : $ python3 run.py --tflite BinaryModel.tflite --source path/to/image or path/to/directoryImage \
                    --classesNames  path/to/FileClasseName.txt --imgsz 224 --channels 3
"""


IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def run(tflite,source,classesNames,imgsz=224,channels=3):

    #to load classes names
    if not (os.path.isfile(classesNames)):
        return "argument classesNames must be a file"
    df = pd.DataFrame(columns=['Image', 'ClassName' ,'Confidence','Inference time'])
    with open(classesNames) as labels:
         classes = [i.strip() for i in labels.readlines()]
    n = len(classes) # number of classe


    is_file = Path(source).suffix[1:] in IMG_FORMATS
    results = {}
    if is_file :
        result = LCE.lce(str(tflite),str(source),str(classesNames),int(n),int(imgsz),int(channels))
        results = json.loads(result)
        df = df.append(results,ignore_index=True)
    elif os.path.isdir(source):
        for f in os.listdir(source):
            img = os.path.join(source,f)
            result = LCE.lce(str(tflite),str(img),str(classesNames),int(n))
            results = json.loads(result)
            df = df.append(results,ignore_index=True)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tflite', type=str, default=ROOT / 'BinaryAlexNet.tflite', help='model tflite path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'test.jpg', help='file/dir/URL/glob')
    parser.add_argument('--classesNames', type=str, default=ROOT / 'imagenet_label.txt', help='/path/to/FileClasseName')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=224, help='inference size h,w')
    parser.add_argument('--channels', type=float, default=0.25, help='number channel')
    opt = parser.parse_args()
    df = run(**vars(opt))
    print(df)
