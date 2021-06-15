import os
from utils import chair 
import argparse
    
parser = argparse.ArgumentParser()
parser.add_argument("--annotation_path", type=str, default='coco/annotations')
parser.add_argument("--result_path", type=str, default='example_generated_sentences/trm_karpathy_coco_split_beam2_rl_ep16.json')
args = parser.parse_args()

_, imids, _ = chair.load_generated_captions(args.result_path)

evaluator = chair.CHAIR(imids, args.annotation_path) 
evaluator.get_annotations()

result_name = os.path.basename(args.result_path).split('.')[0]
print("\t\t {} \t\t".format(result_name))

cap_dict = evaluator.compute_chair(args.result_path) 
metric_string = chair.print_metrics(cap_dict, False)
chair.save_hallucinated_words(args.result_path, cap_dict)
