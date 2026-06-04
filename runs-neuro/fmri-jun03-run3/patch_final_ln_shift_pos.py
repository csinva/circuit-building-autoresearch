import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

content = content.replace(
    "model.final_ln.bias.data += 1.18",
    "model.final_ln.bias.data += 1.18\n        model.final_ln.bias.data[1001] = 0.0" # remove shift from time dim
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Final_LN_Shift_Time_0")

with open(filepath, "w") as f:
    f.write(content)
print("Applied time 0 LN shift patch")
