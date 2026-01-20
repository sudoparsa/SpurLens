from typing import List, Any, Callable, Union
import os
import random
import numpy as np
import pickle as pkl
from compute_rankings import extract_feature_score_owl, extract_feature_score_dino

if __name__ == '__main__':
	from env_vars import CACHE_DIR, PIPELINE_STORAGE_DIR, COCO_PATH
	from image_mask_datasets import get_image_mask_dataset
	import math
	import gc
	import pickle as pkl
	import argparse
	import json
	from collections import defaultdict
	import os
	import pathlib
	from utils import format_name

	parser = argparse.ArgumentParser(description="Ranking Dataset by Object Detection Scores")
	parser.add_argument(
		"--coco_cls",
		type=int,
		help="COCO class for spurious feature list",
		required=True
	)
	parser.add_argument(
		"--model",
		type=str,
		choices=['owl', 'dino'],
		help="Object detection model to use",
		required=True
	)
	args = parser.parse_args()
	cls = args.coco_cls
	model_name = args.model
	spur_feat_file = f"coco-{cls}.txt"

	if model_name == 'owl':
		extract_feat_score = extract_feature_score_owl
	elif model_name == 'dino':
		extract_feat_score = extract_feature_score_dino
	else:
		raise Exception(f"Model '{model_name}' is not supported")
	
	split = 'val'
	annotation_file = os.path.join(COCO_PATH, 'annotations', f'instances_{split}2017.json')
	with open(annotation_file, 'r') as f:
		instances_data = json.load(f)
	
	supercats = defaultdict(list)
	for d in instances_data['categories']:
		supercats[d['supercategory']].append(d['id'])
	supercats = dict(supercats)

	id_to_supercat = {}
	for d in instances_data['categories']:
		id_to_supercat[d['id']] = d['supercategory']

	spur_feat_file = f"coco-{cls}.txt"
	with open(os.path.join(PIPELINE_STORAGE_DIR, 'spurious_features', spur_feat_file), 'r') as f:
		all_spur_features = [line.strip() for line in f.readlines()]
	
	spurious_dataset = get_image_mask_dataset(f"coco-{cls}")
	spurious_image_ids = set(spurious_dataset.image_id_lst)
	
	spur_feat_to_big_lst = {}
	for spur_feat_idx in range(len(all_spur_features)):
		big_lst = []
		for other_cls in supercats[id_to_supercat[cls]]:
			if cls == other_cls:
				continue
			images_dataset = get_image_mask_dataset(f"coco-{other_cls}")
			results_dir = os.path.join(PIPELINE_STORAGE_DIR, 'cross_object_detection', f"coco-{cls}", f"coco-{other_cls}", 'owl')

			# results = [os.path.join(results_dir, fname) for fname in os.listdir(results_dir)]
			all_idxs = [i for i in range(len(images_dataset)) if images_dataset.image_id_lst[i] not in spurious_image_ids]
			results = [os.path.join(results_dir, f"{i}.pkl") for i in all_idxs]

			for i, res_path in enumerate(results):
				with open(res_path, 'rb') as f:
					res = pkl.load(f)
				score = extract_feature_score_owl(res, spur_feat_idx)
				j = int(res_path.split('/')[-1].split('.')[0])
				big_lst.append((score, other_cls, j))
		random.shuffle(big_lst)
		big_lst.sort(key=lambda t: t[0])
		spur_feat_to_big_lst[all_spur_features[spur_feat_idx]] = [(t[1], t[2], t[0]) for t in big_lst]
	
	output_file = os.path.join(PIPELINE_STORAGE_DIR, 'cross_rankings', f"coco-{cls}.pkl")
	with open(output_file, 'wb') as f:
		pkl.dump(dict(spur_feat_to_big_lst), f)
	