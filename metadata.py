import os
import json


test_set = [5, 7, 8, 11, 12, 13, 15, 16, 20, 21, 23, 24, 27, 29, 32, 36, 40, 41, 42, 45, 47, 48, 49, 55, 56, 63, 64, 66, 72, 73, 76, 78, 79, 81, 82, 83, 84, 86, 87, 88, 91, 94, 95, 96, 97, 98, 99, 101, 113, 125, 129, 130]

isolated = [i for i in range(6, 11)] + [i for i in range(41, 71)] + [i for i in range(101, 131)]

for id in range(1, 131):
    with open(os.path.join('data', 'scene{}'.format(id), 'metadata.json'), 'r') as fp:
        meta = json.load(fp)
    if id in test_set:
        meta["split"] = "test"
    else:
        meta["split"] = "train"
    if id in isolated:
        meta["type"] = "isolated"
    else:
        meta["type"] = "cluttered"
    with open(os.path.join('data', 'scene{}'.format(id), 'metadata.json'), 'w') as fp:
        json.dump(meta, fp)
