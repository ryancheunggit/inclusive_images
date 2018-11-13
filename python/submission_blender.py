from CONSTANTS import *
import pandas as pd
from tqdm import tqdm

# sub1 = pd.read_csv(SUB_DIR + 'ResNet101_dual_1024_iteration_90_combined.csv') # PLB 0.471
# sub2 = pd.read_csv(SUB_DIR + 'ResNet50_iteration_144_ATT_added_by_priori.csv') # PLB 0.482
sub1 = pd.read_csv(SUB_DIR + 'ResNet101_dual_1024_iteration_90_tuned_f20.289.csv') # PLB 0.288
sub2 = pd.read_csv(SUB_DIR + 'ResNet50_iteration_144_tuned_f20.282.csv') # PLB 0.280
sub1.fillna('', inplace=True)
sub2.fillna('', inplace=True)


num_common = []
num_total = []
sub_int = sub1.copy()
sub_union = sub1.copy()
for i in tqdm(range(sub1.shape[0])):
    pred1 = set(sub1.iloc[i, 1].split())
    pred2 = set(sub2.iloc[i, 1].split())
    intersection = pred1.intersection(pred2)
    num_common.append(len(intersection))
    union = pred1.union(pred2)
    num_total.append(len(union))
    sub_int.iloc[i, 1] = ' '.join(intersection)
    sub_union.iloc[i, 1] = ' '.join(union)

diff = list(map(lambda x: x[1] - x[0], zip(num_common, num_total)))
ratio = list(map(lambda x: x[0] / (x[1] + 1e-7), zip(num_common, num_total)))

# sub_int.to_csv(SUB_DIR + '50_144_cb_intersect_blend_101_90_cb.csv', index=False)
# sub_union.to_csv(SUB_DIR + '50_144_cb_union_blend_101_90_cb.csv', index=False)
sub_int.to_csv(SUB_DIR + '50_144_raw_intersect_blend_101_90_raw.csv', index=False)
sub_union.to_csv(SUB_DIR + '50_144_raw_union_blend_101_90_raw.csv', index=False)
