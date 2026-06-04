import argparse
import sys
import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"

# Restore to 0.0411 Master
os.system("python3 runs-neuro/fmri-jun03-run3/patch_master_0411.py")

with open(filepath, "r") as f:
    content = f.read()

# We broke 0.0421 by adding a massive positive bias (+1.18) to the final LayerNorm.
# This forces the features into a completely different activation regime right before the ridge regression.
content = content.replace(
    "nn.init.zeros_(model.final_ln.bias)",
    "nn.init.zeros_(model.final_ln.bias)\n        model.final_ln.bias.data += 1.18"
)

content = content.replace("Deep_Ensemble_0411_Master", "Deep_Ensemble_0421_Master")
content = content.replace("pure exact token routing in 960-988, and tuned L2 splits (1.15x/1.0x/0.85x).", "pure exact token routing in 960-988, tuned L2 splits (1.15x/1.0x/0.85x), and final LN bias +1.18.")

with open(filepath, "w") as f:
    f.write(content)

os.system("cp runs-neuro/fmri-jun03-run3/interpretable_transformer.py runs-neuro/fmri-jun03-run3/final_model_0421.py")
print("Updated successfully to 0.0421")
