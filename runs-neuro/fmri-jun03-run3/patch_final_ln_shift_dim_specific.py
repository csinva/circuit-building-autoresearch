import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Instead of shifting everything by 1.18, let's shift everything by 1.18 EXCEPT the exact tokens (960-988)
# Or just shift the exact tokens differently
content = content.replace(
    "model.final_ln.bias.data += 1.18",
    "model.final_ln.bias.data += 1.18\n        model.final_ln.bias.data[960:988] = 0.0"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Final_LN_Shift_Dim_Specific")

with open(filepath, "w") as f:
    f.write(content)
print("Applied dim specific LN shift patch")
