from typing import List, Any, Callable, Union
import os
import random
import numpy as np
import pickle as pkl
from compute_rankings import extract_feature_score_owl, extract_feature_score_dino
from env_vars import IMAGENET_PATH
from image_mask_datasets import CachedImageNet, ImagenetCategoryImageMaskDataset

if __name__ == '__main__':
	from env_vars import CACHE_DIR, PIPELINE_STORAGE_DIR
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
		"--imagenet_cls",
		type=int,
		help="Imagenet class for spurious feature list",
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
	cls = args.imagenet_cls
	model_name = args.model
	spur_feat_file = f"imagenet-{cls}.txt"

	if model_name == 'owl':
		extract_feat_score = extract_feature_score_owl
	elif model_name == 'dino':
		extract_feat_score = extract_feature_score_dino
	else:
		raise Exception(f"Model '{model_name}' is not supported")
	
	cin = CachedImageNet(root=IMAGENET_PATH, split='train')
	with open('<<REDACTED: Spurious Imagenet path>>/included_classes.txt', 'r') as f:
		available_classes_lst = [int(line.strip()) for line in f.readlines()]
	datasets = {}
	for other_cls in available_classes_lst:
		if other_cls == cls:
			continue
		datasets[other_cls] = ImagenetCategoryImageMaskDataset(other_cls, 'train', cin)

	spur_feat_file = f"imagenet-{cls}.txt"
	with open(os.path.join(PIPELINE_STORAGE_DIR, 'spurious_features', spur_feat_file), 'r') as f:
		all_spur_features = [line.strip() for line in f.readlines()]
	
	objdet_dir = os.path.join(PIPELINE_STORAGE_DIR, 'cross_object_detection', f"imagenet-{cls}.txt", 'owl')
	main_lst = []
	for filename in os.listdir(objdet_dir):
		x, y = filename.split('.')[0].split('_')
		try:
			with open(os.path.join(objdet_dir, filename), 'rb') as f:
				obj = pkl.load(f)
		except:
			print(f"FAILED TO OPEN {x=}, {y=}")
		main_lst.append((obj, x, y))
	
	spur_feat_to_big_lst = {}
	for spur_feat_idx in range(len(all_spur_features)):
		random.shuffle(main_lst)
		score_lst = [(extract_feature_score_owl(res, spur_feat_idx), x, y) for (res, x, y) in main_lst]
		score_lst.sort(key=lambda t: t[0])
		spur_feat_to_big_lst[all_spur_features[spur_feat_idx]] = [(t[1], t[2], t[0]) for t in score_lst]
	
	output_file = os.path.join(PIPELINE_STORAGE_DIR, 'cross_rankings', f"imagenet-{cls}.pkl")
	with open(output_file, 'wb') as f:
		pkl.dump(dict(spur_feat_to_big_lst), f)
	