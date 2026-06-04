import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ManualInterpretableTransformer(nn.Module):
    def __init__(self, vocab_size=256, d_model=1024, n_heads=16, d_ff=4096, max_seq_len=64, n_layers=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_seq_len = max_seq_len
        self.d_k = d_model // n_heads
        self.n_layers = n_layers
        
        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        
        # Randomize token embeddings
        nn.init.normal_(self.token_emb.weight, std=1.0)
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'q_proj': nn.Linear(d_model, d_model, bias=False),
                'k_proj': nn.Linear(d_model, d_model, bias=False),
                'v_proj': nn.Linear(d_model, d_model, bias=False),
                'o_proj': nn.Linear(d_model, d_model, bias=False),
                'ff1': nn.Linear(d_model, d_ff),
                'ff2': nn.Linear(d_ff, d_model),
                'ln1': nn.LayerNorm(d_model),
                'ln2': nn.LayerNorm(d_model)
            }) for _ in range(n_layers)
        ])
        
        self._initialize_as_cnn()

    def _initialize_as_cnn(self):
        """
        Initializes the Transformer to act EXACTLY as a 1D Convolutional Neural Network.
        - Q, K projections are zeroed (attention relies strictly on relative positional biases).
        - Relative Positional biases are forced to hard-shift by `h` steps for head `h`.
        - V, O projections apply random orthogonal mixing.
        - MLP applies random mixing with ReLU.
        """
        for i, layer in enumerate(self.layers):
            # Zero out Q and K to ignore content for attention
            nn.init.zeros_(layer['q_proj'].weight)
            nn.init.zeros_(layer['k_proj'].weight)
            
            # V projection: Identity mapping or random orthogonal
            nn.init.orthogonal_(layer['v_proj'].weight, gain=1.0)
            
            # O projection: Random orthogonal
            nn.init.orthogonal_(layer['o_proj'].weight, gain=1.0)
            
            # FF networks: Random orthogonal weights to preserve variance
            nn.init.orthogonal_(layer['ff1'].weight, gain=np.sqrt(2))
            nn.init.zeros_(layer['ff1'].bias)
            nn.init.orthogonal_(layer['ff2'].weight, gain=1.0)
            nn.init.zeros_(layer['ff2'].bias)
            
            # Normalization biases zero
            nn.init.zeros_(layer['ln1'].bias)
            nn.init.ones_(layer['ln1'].weight)
            nn.init.zeros_(layer['ln2'].bias)
            nn.init.ones_(layer['ln2'].weight)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        x = self.token_emb(input_ids)
        
        # Build strict CNN-like shift relative biases
        # bias[h, i, j] = 0 if (i - j) == h else -inf
        positions = torch.arange(seq_len, device=input_ids.device)
        # diff[i, j] = i - j  (positive means past)
        diff = positions.unsqueeze(1) - positions.unsqueeze(0)
        
        # head index h maps to a shift of h
        # for h in [0..n_heads-1]
        h_idx = torch.arange(self.n_heads, device=input_ids.device).view(self.n_heads, 1, 1)
        
        # We want diff == h_idx. But wait, if diff < 0 (future), it's never == h_idx (which are >= 0).
        # What if a shift asks for a token before the start?
        # The attention row will have all -inf. Softmax will yield uniform.
        # Let's fix that by adding a dummy position or just letting it be 0 if valid, -10000 otherwise.
        match = (diff.unsqueeze(0) == h_idx)
        rel_bias = torch.where(match, torch.zeros(1, device=x.device), torch.full((1,), -10000.0, device=x.device))
        
        for layer_idx, layer in enumerate(self.layers):
            # Self-attention block
            residual = x
            x_norm = layer['ln1'](x)
            
            q = layer['q_proj'](x_norm).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            k = layer['k_proj'](x_norm).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            v = layer['v_proj'](x_norm).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
            
            # Attention with fixed relative biases
            attn_weights = (q @ k.transpose(-2, -1)) / np.sqrt(self.d_k)
            attn_weights = attn_weights + rel_bias.unsqueeze(0)
            
            attn_probs = F.softmax(attn_weights, dim=-1)
            # If a row was all -10000 (meaning the required shift is out of bounds), softmax makes it uniform.
            # We want it to be all zeros. 
            # We can mask it:
            valid_mask = match.unsqueeze(0).float()
            attn_probs = attn_probs * valid_mask
            
            attn_out = (attn_probs @ v).transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
            attn_out = layer['o_proj'](attn_out)
            
            x = residual + attn_out
            
            # MLP block
            residual = x
            x_norm = layer['ln2'](x)
            ff_out = layer['ff2'](F.relu(layer['ff1'](x_norm)))
            x = residual + ff_out

        return x

def get_model():
    return ManualInterpretableTransformer(n_layers=4, d_model=1024, n_heads=16, d_ff=4096)

if __name__ == "__main__":
    import sys
    from src.eval import EncodingConfig, run_encoding, make_result_row, upsert_overall_results
    import os
    
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = get_model().to(device)
    
    class InterpretableEmbedder:
        def __init__(self, model, device='cpu'):
            self.model = model
            self.device = device
            
            # Simple vocab mapping for characters
            chars = " abcdefghijklmnopqrstuvwxyz.,;:'\"!?-_=()[]{}0123456789"
            self.vocab = {c: i for i, c in enumerate(chars)}
            self.vocab_size = len(self.vocab)

        def __call__(self, texts: list[str]) -> np.ndarray:
            features = []
            for text in texts:
                # Truncate or pad text to max_seq_len (we process in chunks usually, but here text is already chunked)
                seq_len = self.model.max_seq_len
                # In feature extraction, it passes n-grams.
                # Actually wait, the `texts` here are 10-grams (strings of length 10 or whatever ngram_size is)
                # Let's pad it to max_seq_len
                
                input_ids = torch.zeros((1, seq_len), dtype=torch.long, device=self.device)
                for i, char in enumerate(text[:seq_len]):
                    input_ids[0, i] = self.vocab.get(char.lower(), 0)
                
                with torch.no_grad():
                    out = self.model(input_ids)
                
                # We want the embedding for the LAST character of the sequence, or pooling?
                # The manual transformer used the output of the last valid token.
                # Since texts are exactly ngram_size length, we use out[0, len(text)-1]
                idx = min(len(text)-1, seq_len-1)
                feat = out[0, idx].cpu().numpy()
                features.append(feat)
            return np.array(features)

    embedder = InterpretableEmbedder(model, device=device)
    
    config = EncodingConfig(
        subject="UTS03",
        num_train=8,
        num_test=2,
        ngram_size=10,
        ndelays=4,
        nboots=5,
        chunklen=40,
        nchunks=20,
        trim_edges=True
    )
    
    model_name = "Random_1D_CNN_Transformer"
    print(f"\n--- Testing model: {model_name} ---")
    results = run_encoding(embedder, config)
    
    n_params = sum(p.numel() for p in model.parameters())
    row = make_result_row(results, model_name, n_params, "Pure random 1D Convolutional Neural Network implemented via relative attention biases.")
    upsert_overall_results([row], RESULTS_DIR)
