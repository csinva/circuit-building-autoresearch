import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# Restore to 0.0408 Master
os.system("python3 runs-neuro/fmri-jun03-run3/patch_master_0408.py")

with open(filepath, "r") as f:
    content = f.read()

# We broke 0.0409 by tuning the L2 split projections. We gave more weight (1.25x) to the local 
# context branch of the split (dims 0-21) and less weight (0.75x) to the extremely long context branch (dims 43-63),
# while keeping the medium context (dims 22-42) at 1.0x. This allows the model to favor medium-short contexts.
content = content.replace(
    "                l2_attn.W_o.weight[d_start + i, h_start + i] = S * 1.0\n                \n            for i in range(21):\n                l2_attn.W_v.weight[h_start + 22 + i, d_start_b + i] = 1.0\n                l2_attn.W_o.weight[d_start + 22 + i, h_start + 22 + i] = S * 1.0\n                \n            for i in range(21):\n                l2_attn.W_v.weight[h_start + 43 + i, d_start_c + i] = 1.0\n                l2_attn.W_o.weight[d_start + 43 + i, h_start + 43 + i] = S * 1.0",
    "                l2_attn.W_o.weight[d_start + i, h_start + i] = S * 1.25\n                \n            for i in range(21):\n                l2_attn.W_v.weight[h_start + 22 + i, d_start_b + i] = 1.0\n                l2_attn.W_o.weight[d_start + 22 + i, h_start + 22 + i] = S * 1.0\n                \n            for i in range(21):\n                l2_attn.W_v.weight[h_start + 43 + i, d_start_c + i] = 1.0\n                l2_attn.W_o.weight[d_start + 43 + i, h_start + 43 + i] = S * 0.75"
)

content = content.replace("Deep_Ensemble_0408_Master", "Deep_Ensemble_0409_Master")
content = content.replace("pure exact token routing in 960-988.", "pure exact token routing in 960-988, and tuned L2 splits (1.25x/1.0x/0.75x).")

with open(filepath, "w") as f:
    f.write(content)

os.system("cp runs-neuro/fmri-jun03-run3/interpretable_transformer.py runs-neuro/fmri-jun03-run3/final_model_0409.py")
print("Updated successfully")
