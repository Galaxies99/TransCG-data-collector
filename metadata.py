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

train_set = []
cluttered = []
for id in range(1, 131):
    if id not in test_set:
        train_set.append(id)
    if id not in isolated:
        cluttered.append(id)

train_D435_cnt = 0
train_L515_cnt = 0
test_D435_cnt = 0
test_L515_cnt = 0
isolated_D435_cnt = 0
isolated_L515_cnt = 0
cluttered_D435_cnt = 0
cluttered_L515_cnt = 0

for scene_id in range(1, 131):
    if scene_id in train_set:
        with open(os.path.join('data', 'scene{}'.format(scene_id), "metadata.json"), 'r') as fp:
            meta = json.load(fp)
            train_D435_cnt += meta['D435_valid_perspective_num']
            train_L515_cnt += meta['L515_valid_perspective_num']
    else:
        with open(os.path.join('data', 'scene{}'.format(scene_id), "metadata.json"), 'r') as fp:
            meta = json.load(fp)
            test_D435_cnt += meta['D435_valid_perspective_num']
            test_L515_cnt += meta['L515_valid_perspective_num']
    if scene_id in isolated:
        with open(os.path.join('data', 'scene{}'.format(scene_id), "metadata.json"), 'r') as fp:
            meta = json.load(fp)
            isolated_D435_cnt += meta['D435_valid_perspective_num']
            isolated_L515_cnt += meta['L515_valid_perspective_num']
    else:
        with open(os.path.join('data', 'scene{}'.format(scene_id), "metadata.json"), 'r') as fp:
            meta = json.load(fp)
            cluttered_D435_cnt += meta['D435_valid_perspective_num']
            cluttered_L515_cnt += meta['L515_valid_perspective_num']

assert train_D435_cnt + test_D435_cnt == isolated_D435_cnt + cluttered_D435_cnt
assert train_L515_cnt + test_L515_cnt == isolated_L515_cnt + cluttered_L515_cnt

meta = {
    "total_samples": train_D435_cnt + train_L515_cnt + test_D435_cnt + test_L515_cnt,
    "total_D435_samples": train_D435_cnt + test_D435_cnt,
    "total_L515_samples": train_L515_cnt + test_L515_cnt,
    "train": train_set,
    "train_samples": train_D435_cnt + train_L515_cnt,
    "train_D435_samples": train_D435_cnt,
    "train_L515_samples": train_L515_cnt,
    "test": test_set,
    "test_samples": test_D435_cnt + test_L515_cnt,
    "test_D435_samples": test_D435_cnt,
    "test_L515_samples": test_L515_cnt,
    "isolated": isolated,
    "isolated_samples": isolated_D435_cnt + isolated_L515_cnt,
    "isolated_D435_samples": isolated_D435_cnt,
    "isolated_L515_samples": isolated_L515_cnt,
    "cluttered": cluttered,
    "cluttered_samples": cluttered_D435_cnt + cluttered_L515_cnt,
    "cluttered_D435_samples": cluttered_D435_cnt,
    "cluttered_L515_samples": cluttered_L515_cnt 
}

with open(os.path.join('data', 'metadata.json'), 'w') as fp:
    json.dump(meta, fp)
