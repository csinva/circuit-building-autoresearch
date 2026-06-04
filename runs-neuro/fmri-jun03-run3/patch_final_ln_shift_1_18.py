import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0411.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

content = content.replace(
    "nn.init.zeros_(model.final_ln.bias)",
    "nn.init.zeros_(model.final_ln.bias)\n        model.final_ln.bias.data += 1.18"
)
content = content.replace("Deep_Ensemble_0411_Master", "Deep_Ensemble_Final_LN_Shift_1_18")

with open(filepath, "w") as f:
    f.write(content)
print("Applied final LN shift 1.18 patch")
