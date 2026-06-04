import os

filepath = "runs-neuro/fmri-jun03-run3/interpretable_transformer.py"
os.system("cp runs-neuro/fmri-jun03-run3/final_model_0421.py " + filepath)

with open(filepath, "r") as f:
    content = f.read()

# Change n_layers to 3 in builder
content = content.replace("n_layers: int = 2", "n_layers: int = 3")

# We need to initialize the weights for the 3rd layer
old_init = """        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)"""

new_init = """        l3_attn = model.blocks[2].attn
        mlp3 = model.blocks[2].mlp
        
        for net in range(num_nets):
            d_start = net * dim_per_net
            f_start = net * ff_per_net
            h_start = (net % n_heads) * d_head
            
            # --- LAYER 3: Slower Integration ---
            l3_decay = 0.001 + float(net) * (2.999 / 14.0)
            
            l3_attn.W_q.weight[h_start + 0, d_start + 29] = 5.0
            l3_attn.W_k.weight[h_start + 0, d_start + 28] = l3_decay
            
            # 3-way split again but slower
            net_b = (net + 5) % 15
            net_c = (net + 10) % 15
            d_start_b = net_b * dim_per_net
            d_start_c = net_c * dim_per_net
            
            for i in range(22):
                l3_attn.W_v.weight[h_start + i, d_start + i] = 1.0
                l3_attn.W_o.weight[d_start + i, h_start + i] = S * 1.15
                
            for i in range(21):
                l3_attn.W_v.weight[h_start + 22 + i, d_start_b + i] = 1.0
                l3_attn.W_o.weight[d_start + 22 + i, h_start + 22 + i] = S * 1.0
                
            for i in range(21):
                l3_attn.W_v.weight[h_start + 43 + i, d_start_c + i] = 1.0
                l3_attn.W_o.weight[d_start + 43 + i, h_start + 43 + i] = S * 0.85
                
            std_dev3 = 0.01 + float(net) * (1.99 / 14.0)
            
            nn.init.normal_(mlp3.fc1.weight[f_start:f_start+ff_per_net, d_start:d_start+dim_per_net], std=std_dev3)
            nn.init.normal_(mlp3.fc2.weight[d_start:d_start+dim_per_net, f_start:f_start+ff_per_net], std=std_dev3 * S)
            
        model.blocks[2].mlp.fc1.weight.data[:, 960:988] = 0
        model.blocks[2].mlp.fc2.weight.data[960:988, :] = 0

        for block in model.blocks:
            nn.init.ones_(block.ln1.weight)
            nn.init.zeros_(block.ln1.bias)
            nn.init.ones_(block.ln2.weight)
            nn.init.zeros_(block.ln2.bias)"""

if old_init not in content:
    print("Error: Could not find old init logic to replace.")
    exit(1)

content = content.replace(old_init, new_init)
content = content.replace("Deep_Ensemble_0421_Master", "Deep_Ensemble_3_Layer")

with open(filepath, "w") as f:
    f.write(content)
print("Applied 3-layer patch")
