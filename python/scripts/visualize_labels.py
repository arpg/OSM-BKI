import sys
import os

import composite_bki_cpp
import numpy as np
from visualize_osm import MCD_LABEL_COLORS, MCD_LABEL_NAMES, KITTI_LABEL_COLORS, KITTI_LABEL_NAMES

points = np.fromfile('example_data/mcd_scan/data/0000000011.bin', dtype=np.float32).reshape(-1,4)
labels = np.fromfile('example_data/mcd_scan/pred_labels/0000000011_prediction.label', dtype=np.uint32)

bki = composite_bki_cpp.PyContinuousBKI('example_data/mcd_scan/kth_day_06_osm_geometries.bin', 'configs/mcd_config.yaml')
bki.update(labels, points[:,:3])

print('Grid size after update:', bki.get_size())

preds = bki.infer(points[:,:3])
unique, counts = np.unique(preds, return_counts=True)
print('\nPredicted label distribution:')
for u, c in zip(unique, counts):
    name = MCD_LABEL_NAMES.get(u, 'unknown')
    color = MCD_LABEL_COLORS.get(u, [0,0,0])
    r, g, b = int(color[0]*255), int(color[1]*255), int(color[2]*255)
    color_block = f"\033[48;2;{r};{g};{b}m    \033[0m"
    print(f'  Label {u:<2} ({name:<15}): {c:<6} points ({100*c/len(preds):.1f}%) {color_block}')
    
# Check the original labels
unique_orig, counts_orig = np.unique(labels, return_counts=True)
print('\nOriginal label distribution:')
for u, c in zip(unique_orig, counts_orig):
    name = MCD_LABEL_NAMES.get(u, 'unknown')
    color = MCD_LABEL_COLORS.get(u, [0,0,0])
    r, g, b = int(color[0]*255), int(color[1]*255), int(color[2]*255)
    color_block = f"\033[48;2;{r};{g};{b}m    \033[0m"
    print(f'  Label {u:<2} ({name:<15}): {c:<6} points ({100*c/len(labels):.1f}%) {color_block}')
