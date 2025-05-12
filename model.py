import torch
import torch.nn as nn
import math
import config

class PositionalEncoding(nn.Module):
    """ Standard positional encoding for Transformers. """
    def __init__(self, d_model, max_len=500): # Adjust max_len if needed
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        # Add positional encoding up to the length of the input sequence
        x = x + self.pe[:, :x.size(1)]
        return x

class PathfindingTransformer(nn.Module):
    """ Transformer model for pathfinding using local patches. """
    def __init__(self, grid_vocab_size, coord_vocab_size, embed_dim, num_heads,
                 num_layers, d_ff, num_actions, patch_len, max_seq_len, dropout=0.1):
        """
        Args:
            grid_vocab_size (int): Size of grid cell vocabulary (e.g., 2 for free/obstacle).
            coord_vocab_size (int): Max coordinate value + 1.
            embed_dim (int): Embedding dimension.
            num_heads (int): Number of attention heads.
            num_layers (int): Number of transformer encoder layers.
            d_ff (int): Dimension of the feed-forward layer.
            num_actions (int): Number of possible output actions.
            patch_len (int): Length of the flattened patch (patch_size * patch_size).
            max_seq_len (int): Maximum sequence length (patch_len + 2).
            dropout (float): Dropout rate.
        """
        super(PathfindingTransformer, self).__init__()
        self.embed_dim = embed_dim
        self.patch_len = patch_len

        # --- Embeddings ---
        # Embedding for grid cells in the patch
        self.grid_cell_embedding = nn.Embedding(grid_vocab_size, embed_dim, padding_idx=None) # Assuming no specific pad token needed here

        # Embeddings for row and column coordinates (concatenated for position)
        # Ensure coord_vocab_size is large enough for max grid dimension (e.g., 100)
        self.row_embedding = nn.Embedding(coord_vocab_size, embed_dim // 2)
        self.col_embedding = nn.Embedding(coord_vocab_size, embed_dim // 2)

        # Positional Encoding for the sequence (patch + current_pos + goal_pos)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_seq_len)

        # --- Transformer Encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True # Expect input as (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # --- Output Layer ---
        # Pool the transformer outputs and then apply a linear layer
        # Alternative: Use output corresponding to a special token if added (e.g., CLS)
        self.output_layer = nn.Linear(embed_dim, num_actions)

        self.dropout = nn.Dropout(dropout)

    def _embed_pos(self, pos_tensor):
        """Embeds (row, col) tensor to a single embedding vector."""
        # pos_tensor shape: (batch, 2) where pos_tensor[:, 0] is row, pos_tensor[:, 1] is col
        r_emb = self.row_embedding(pos_tensor[:, 0]) # (batch, embed_dim // 2)
        c_emb = self.col_embedding(pos_tensor[:, 1]) # (batch, embed_dim // 2)
        # Concatenate row and column embeddings
        pos_emb = torch.cat((r_emb, c_emb), dim=-1) # (batch, embed_dim)
        return pos_emb

    def forward(self, flat_patch_input, current_pos_input, goal_pos_input):
        """
        Args:
            flat_patch_input (Tensor): Flattened local patch. Shape: (batch, patch_len).
            current_pos_input (Tensor): Current position (r, c). Shape: (batch, 2).
            goal_pos_input (Tensor): Goal position (r, c). Shape: (batch, 2).

        Returns:
            Tensor: Action logits. Shape: (batch, num_actions).
        """
        # 1. Embed Inputs
        patch_emb = self.grid_cell_embedding(flat_patch_input) # (batch, patch_len, embed_dim)
        current_pos_emb = self._embed_pos(current_pos_input).unsqueeze(1) # (batch, 1, embed_dim)
        goal_pos_emb = self._embed_pos(goal_pos_input).unsqueeze(1)       # (batch, 1, embed_dim)

        # 2. Concatenate Embeddings into a single sequence
        # Order: [Patch Embeddings, Current Pos Embedding, Goal Pos Embedding]
        full_sequence_emb = torch.cat([patch_emb, current_pos_emb, goal_pos_emb], dim=1)
        # Expected shape: (batch, patch_len + 2, embed_dim)

        # Apply dropout to embeddings (optional but common)
        full_sequence_emb = self.dropout(full_sequence_emb)

        # 3. Add Positional Encoding
        full_sequence_emb = self.pos_encoder(full_sequence_emb)

        # 4. Pass through Transformer Encoder
        # No mask needed for standard encoder unless using padding within sequence
        transformer_output = self.transformer_encoder(full_sequence_emb)
        # Shape: (batch, patch_len + 2, embed_dim)

        # 5. Pool Output & Final Linear Layer
        # Average pooling across the sequence length dimension
        pooled_output = transformer_output.mean(dim=1) # Shape: (batch, embed_dim)

        # Alternative: Use the output of a specific token, e.g., the current_pos token output
        # current_pos_output = transformer_output[:, self.patch_len] # Output corresponding to current_pos_emb
        # action_logits = self.output_layer(current_pos_output)

        action_logits = self.output_layer(pooled_output) # Shape: (batch, num_actions)

        return action_logits

# --- Function to create model instance ---
def create_model():
    model = PathfindingTransformer(
        grid_vocab_size=config.GRID_VOCAB_SIZE,
        coord_vocab_size=config.COORD_VOCAB_SIZE,
        embed_dim=config.EMBED_DIM,
        num_heads=config.NUM_HEADS,
        num_layers=config.NUM_LAYERS,
        d_ff=config.D_FF,
        num_actions=config.NUM_ACTIONS,
        patch_len=config.PATCH_SIZE * config.PATCH_SIZE,
        max_seq_len=config.MODEL_MAX_SEQ_LEN,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    return model
