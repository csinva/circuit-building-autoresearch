import os
import re

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

replacement = """
    def forward(self, idx):
        x = self.word_emb(idx)
        pos = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)
        x = x + self.pos_emb(pos)
        
        # Apply L1
        x = self.blocks[0](x)
        
        # Apply L2
        x = self.blocks[1](x)
        
        x = self.final_ln(x)
        return x
"""
# That was just looking at the forward function. Let's do this via parameter manipulation.

content = content.replace(
    "nn.init.zeros_(model.final_ln.bias)",
    "nn.init.zeros_(model.final_ln.bias)\n        model.final_ln.bias.data += 1.18"
)

# wait, we already have model.final_ln.bias.data += 1.18 inside final_model_0421.py.
# So I want to REMOVE that shift from the dimensions mapping to Layer 2 output.
# Layer 1's projection was on dim 30+
# Let's shift ONLY dimensions 0-960 and see what happens? We already tried shifting dim specific and it dropped to 0.0401.

# Let's revert the shift on exact tokens again, but this time with a different value
content = content.replace(
    "model.final_ln.bias.data += 1.18",
    "model.final_ln.bias.data += 1.18\n        model.final_ln.bias.data[960:988] = -0.5"
)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_Exact_Tokens_Neg_0_5")

with open(filepath, "w") as f:
    f.write(content)
print("Applied exact tokens neg bias patch")
