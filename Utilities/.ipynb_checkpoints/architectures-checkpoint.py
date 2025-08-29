from transformers import AutoModel
import torch.nn.functional as F
from transformers import RobertaModel
from torch import nn
import torch
import math

class BERT(nn.Module):
    def __init__(self, encoding_dimension, PRE_TRAINED_MODEL_NAME, n_classes, n_classes2 = None):
        super(BERT, self).__init__()
        self.bert_model = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

        self.encoding_dimension = encoding_dimension
        self.n_classes = n_classes
        self.n_classes_2 = n_classes2

        self.drop_out = nn.Dropout(p = 0.05)

        self.fc1 = nn.Linear(self.encoding_dimension, self.n_classes)
        if self.n_classes_2 is not None:
            self.fc2 = nn.Linear(self.encoding_dimension, self.n_classes_2)

    def forward(self, ids, mask, token_type_ids):
        last_hidden_state, pooler_output = self.bert_model(ids,
                                                           attention_mask = mask,
                                                           token_type_ids = token_type_ids,
                                                           return_dict = False,
                                                           output_hidden_states = False,
                                                           output_attentions = False)
        pooler_drop = self.drop_out(pooler_output)
        logits = self.fc1(pooler_drop)

        if self.n_classes_2 is not None:
            logits2 = self.fc2(self.drop_out(pooler_output))
            return logits, logits2

        return logits
    
class DistilBERT(nn.Module):
    def __init__(self, encoding_dimension, PRE_TRAINED_MODEL_NAME, n_classes, n_classes2 = None):
        super(DistilBERT, self).__init__()
        if PRE_TRAINED_MODEL_NAME == "roberta-base":
            self.distil_bert_model = RobertaModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        else:
            self.distil_bert_model = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

        self.encoding_dimension = encoding_dimension
        self.n_classes = n_classes
        self.n_classes_2 = n_classes2

        self.drop_out = nn.Dropout(p = 0.05)
        self.fc1 = nn.Linear(self.encoding_dimension, self.n_classes)
        if self.n_classes_2 is not None:
            self.fc2 = nn.Linear(self.encoding_dimension, self.n_classes_2)

    def forward(self, ids, mask):
        outputs = self.distil_bert_model(ids,
                                        attention_mask = mask,
                                        return_dict = True,
                                        output_hidden_states = True,
                                        output_attentions = False)
        last_hidden_states  = outputs.hidden_states[-1]
        cls_tensor = last_hidden_states[:,0,:]
        
        cls_drop = self.drop_out(cls_tensor)
        logits = self.fc1(cls_drop)

        if self.n_classes_2 is not None:
            logits2 = self.fc2(self.drop_out(cls_tensor))
            return logits, logits2
        
        return logits

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention (no multi-heads).

    Args:
        Q: Tensor of shape (batch_size, seq_len_q, d_k)
        K: Tensor of shape (batch_size, seq_len_k, d_k)
        V: Tensor of shape (batch_size, seq_len_v, d_v)
        mask: Optional mask of shape (batch_size, seq_len_q, seq_len_k)

    Returns:
        output: Tensor of shape (batch_size, seq_len_q, d_v)
        attention_weights: Tensor of shape (batch_size, seq_len_q, seq_len_k)
    """
    d_k = Q.size(-1)

    # Compute attention scores
    scores = torch.bmm(Q, K.transpose(1, 2)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

    # Apply mask (optional)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Normalize the scores to get attention weights
    attention_weights = F.softmax(scores, dim=-1)

    # Compute the attention output
    output = torch.bmm(attention_weights, V)

    return output, attention_weights

# Create the Dual Attention Layer class (this layer was presented in the paper "Gated Hierarchical Attention")
class ContextualAttentionLayer(nn.Module):
    """
    Applies contextual attention over a sequence using a trainable context vector.

    Args:
        hidden_dim (int): Dimensionality of input encodings.
        dropout_rate (float): Dropout probability.
        batch_first (bool): Whether input shape is (B, L, D).
        return_attention_values (bool): Whether to return attention scores.
    """
    def __init__(self, hidden_dim: int, dropout_rate: float, batch_first: bool, return_attention_values: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.return_attention_values = return_attention_values

        self.sequence_projection = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.context_vector = nn.Linear(hidden_dim, 1, bias=True)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, encoding_sequence: torch.Tensor, att_mask: torch.Tensor, return_context: bool = False):
        """
        Args:
            encoding_sequence (Tensor): Input of shape (B, L, D)
            att_mask (Tensor): Binary mask of shape (B, L)
            return_context (bool): Whether to return raw attention logits
        
        Returns:
            Depending on `return_attention_values` and `return_context`, returns:
            - contextual_representation: (B, L, D)
            - attention_values (optional): (B, L)
            - context_logits (optional): (B, L)
        """
        # Project the sequence and apply non-linearity
        hidden = torch.tanh(self.sequence_projection(encoding_sequence))  # (B, L, D)
        attn_logits = self.context_vector(hidden).squeeze(-1)             # (B, L)

        # Apply mask: set -inf where mask == 0
        attn_logits = attn_logits.masked_fill(att_mask == 0, float('-inf'))

        # Softmax over sequence length
        attn_weights = F.softmax(attn_logits, dim=-1)                     # (B, L)
        attn_weights = self.dropout(attn_weights)

        # Element-wise multiplication (B, L, D)
        contextual_representation = encoding_sequence * attn_weights.unsqueeze(-1)

        outputs = [contextual_representation]
        if self.return_attention_values:
            outputs.append(attn_weights)
        if return_context:
            outputs.append(attn_logits)

        return outputs[0] if len(outputs) == 1 else tuple(outputs)
            
# Create the GetQKV layer
class GetQKV(nn.Module):
    """
    Efficient module to compute Query, Key, and Value matrices from input encodings.
    
    Args:
        hidden_dim (int): The dimensionality of the input and output embeddings.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Use a single linear layer to project to Q, K, V jointly
        self.qkv_projection = nn.Linear(hidden_dim, 3 * hidden_dim, bias=True)

    def forward(self, encoding_sequence: torch.Tensor):
        """
        Args:
            encoding_sequence (Tensor): shape (batch_size, seq_len, hidden_dim)
        
        Returns:
            Tuple[Tensor, Tensor, Tensor]: Q, K, V matrices, each of shape (batch_size, seq_len, hidden_dim)
        """
        # Project and split into Q, K, V
        qkv = self.qkv_projection(encoding_sequence)  # (B, L, 3 * D)
        query, key, value = torch.chunk(qkv, chunks=3, dim=-1)
        return query, key, value

# Create the GAttention Layer
class GAttention(nn.Module):
    def __init__(self, hidden_dim, batch_first=True, device=None, seq_len=None, multi_head=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.batch_first = batch_first
        self.multi_head = multi_head
        self.seq_len = seq_len

        self.Wmod1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.Wmod2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.Wz = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=True)

        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.contextual_attention_layer = ContextualAttentionLayer(
            hidden_dim, dropout_rate=0.05, batch_first=True
        )
        self.qkv = GetQKV(hidden_dim)

    def forward(self, hidden_representation, att_mask, return_sigma=False, return_context=False):
        # Compute self-attention representation
        Q, K, V = self.qkv(hidden_representation)
        sa_rep, attn_values = scaled_dot_product_attention(Q, K, V)

        # Compute contextual attention representation
        ca_rep = self.contextual_attention_layer(Q, att_mask, return_context)

        # Modulate both representations
        h1 = F.relu(self.Wmod1(sa_rep))  # (B, L, D)
        h2 = F.relu(self.Wmod2(ca_rep))  # (B, L, D)

        # Gating mechanism
        x_cat = torch.cat((sa_rep, ca_rep), dim=2)  # (B, L, 2D)
        z = torch.sigmoid(self.Wz(x_cat).sum(dim=2)) * att_mask  # (B, L)

        z_inv = (1 - z) * att_mask

        # Blend representations
        h = h1 * z.unsqueeze(-1) + h2 * z_inv.unsqueeze(-1)  # (B, L, D)

        # Output selection
        outputs = [h]
        if return_sigma:
            outputs.extend([z, z_inv])
        if return_context:
            outputs.append(context_vector)

        return outputs[0] if len(outputs) == 1 else tuple(outputs)

class MultiHeadGAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=12, dropout=0.05, batch_first=True):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim debe ser divisible entre num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.batch_first = batch_first

        # Proyecciones QKV para cada cabeza
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Contextual attention para cada cabeza (reutilizamos una sola capa que procese por cabeza)
        self.contextual_att = ContextualAttentionLayer(
            self.head_dim, dropout_rate=dropout, batch_first=batch_first
        )

        # Modulación por cabeza
        self.Wmod1 = nn.Linear(self.head_dim, self.head_dim)
        self.Wmod2 = nn.Linear(self.head_dim, self.head_dim)
        self.Wz = nn.Linear(self.head_dim * 2, self.head_dim * 2)

        # Salida final
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def split_heads(self, x):
        # (B, L, D) → (B, H, L, D/H)
        B, L, D = x.size()
        return x.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    def combine_heads(self, x):
        # (B, H, L, D/H) → (B, L, D)
        B, H, L, D = x.size()
        return x.transpose(1, 2).contiguous().view(B, L, H * D)

    def scaled_dot_product(self, Q, K, V, mask=None):
        # (B, H, L, D)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, V), attn  # (B, H, L, D), (B, H, L, L)

    def forward(self, hidden_representation, att_mask, return_sigma=False, return_context=False):
        # (B, L, D) → (B, H, L, D/H)
        Q = self.split_heads(self.q_proj(hidden_representation))
        K = self.split_heads(self.k_proj(hidden_representation))
        V = self.split_heads(self.v_proj(hidden_representation))

        # Multi-head self-attention
        sa_rep, _ = self.scaled_dot_product(Q, K, V, att_mask)  # (B, H, L, D/H)

        # Multi-head contextual attention
        # Procesamos cada cabeza por separado
        ca_rep = []
        context_vector = []

        for h in range(self.num_heads):
            Qh = Q[:, h]  # (B, L, D/H)
            ca_out = self.contextual_att(Qh, att_mask, return_context)
            ca_rep.append(ca_out.unsqueeze(1))  # (B, 1, L, D/H)

        ca_rep = torch.cat(ca_rep, dim=1)  # (B, H, L, D/H)

        # Modulación por cabeza
        h1 = F.relu(self.Wmod1(sa_rep))  # (B, H, L, D/H)
        h2 = F.relu(self.Wmod2(ca_rep))  # (B, H, L, D/H)

        # Gating por cabeza
        x_cat = torch.cat((sa_rep, ca_rep), dim=-1)  # (B, H, L, 2D/H)
        z = torch.sigmoid(self.Wz(x_cat).sum(dim=-1)) * att_mask.unsqueeze(1)  # (B, H, L)
        z_inv = (1 - z) * att_mask.unsqueeze(1)

        h = h1 * z.unsqueeze(-1) + h2 * z_inv.unsqueeze(-1)  # (B, H, L, D/H)

        # Combine heads
        h_combined = self.combine_heads(h)  # (B, L, D)
        h_final = self.out_proj(h_combined)  # (B, L, D)

        return h_final

class DistilBERTGAttention(nn.Module):
    def __init__(self, encoding_dimension, PRE_TRAINED_MODEL_NAME, seq_len, n_classes, device, n_classes2 = None):
        super().__init__()

        if PRE_TRAINED_MODEL_NAME == "roberta-base":
            self.distil_bert_model = RobertaModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        else:
            self.distil_bert_model = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME)

        self.encoding_dimension = encoding_dimension
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.n_classes_2 = n_classes2
        self.device = device

        self.dropout = nn.Dropout(p=0.05)

        self.layer_norm_1 = nn.LayerNorm(encoding_dimension)
        self.layer_norm_2 = nn.LayerNorm(encoding_dimension)

        self.gattention_layer = GAttention(encoding_dimension, batch_first=True, device=device, seq_len=seq_len)
        self.feed_forward = nn.Sequential(
            nn.Linear(encoding_dimension, encoding_dimension * 4),
            nn.ReLU(),
            nn.Linear(encoding_dimension * 4, encoding_dimension)
        )

        # Reduce sequence dimension with linear projection after permuting (B, L, D) -> (B, D, L)
        self.fc1 = nn.Linear(seq_len, 1)
        self.fc2 = nn.Linear(encoding_dimension, n_classes)
        if self.n_classes_2 is not None:
            self.fc3 = nn.Linear(encoding_dimension, self.n_classes_2)

    def forward(self, ids, mask, return_sigma=False, return_context=False):
        outputs = self.distil_bert_model(
            input_ids=ids,
            attention_mask=mask,
            return_dict=True,
            output_hidden_states=True,
            output_attentions=False
        )

        last_hidden_state = outputs.hidden_states[-1]  # (B, L, D)

        # Gated attention
        ga_output = self.gattention_layer(last_hidden_state, mask, return_sigma, return_context)

        # Add & Norm after attention
        x = self.layer_norm_1(last_hidden_state + self.dropout(ga_output))  # (B, L, D)

        # Position-wise FFN
        ff_output = self.feed_forward(x)
        x = self.layer_norm_2(x + self.dropout(ff_output))  # (B, L, D)

        # Reduce sequence dimension via learned linear projection
        x_reduced = self.fc1(x.permute(0, 2, 1)).squeeze(-1)  # (B, D)

        logits = self.fc2(self.dropout(x_reduced))  # (B, n_classes)

        if self.n_classes_2 is not None:
            logits2 = self.fc3(self.dropout(x_reduced))
            return logits, logits2

        return logits
    
class BERTGAttention(nn.Module):
    def __init__(self, encoding_dimension, pretrained_model_name, seq_len, n_classes, device, n_classes2 = None):
        super().__init__()

        self.bert_model = AutoModel.from_pretrained(pretrained_model_name)

        self.encoding_dimension = encoding_dimension
        self.n_classes = n_classes
        self.n_classes_2 = n_classes2
        self.device = device

        self.dropout = nn.Dropout(p=0.05)

        self.layer_norm_1 = nn.LayerNorm(encoding_dimension)
        self.layer_norm_2 = nn.LayerNorm(encoding_dimension)

        self.gattention_layer = GAttention(encoding_dimension, batch_first=True, device=device, seq_len=seq_len)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(encoding_dimension, encoding_dimension * 4),
            nn.ReLU(),
            nn.Linear(encoding_dimension * 4, encoding_dimension)
        )

        # This reduces seq_len using a linear projection
        self.fc = nn.Linear(seq_len, 1)  # applied after permuting to (B, D, L)
        self.output_layer = nn.Linear(encoding_dimension, n_classes)
        if self.n_classes_2 is not None:
            self.fc3 = nn.Linear(encoding_dimension, self.n_classes_2)

    def forward(self, ids, mask, token_type_ids, return_sigma=False, return_context=False):
        outputs = self.bert_model(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

        last_hidden_state = outputs.last_hidden_state  # (B, L, D)

        # Gated attention layer
        ga_output = self.gattention_layer(last_hidden_state, mask, return_sigma, return_context)

        # Residual + norm after attention
        x = self.layer_norm_1(last_hidden_state + self.dropout(ga_output))  # (B, L, D)
        
        # Feed-forward network
        ff_output = self.feed_forward(x)
        x = self.layer_norm_2(x + self.dropout(ff_output))  # (B, L, D)
        
        # Reduce sequence using learned linear reduction (transpose to B, D, L)
        x_reduced = self.fc(x.permute(0, 2, 1)).squeeze(-1)  # (B, D)

        logits = self.output_layer(self.dropout(x_reduced))  # (B, n_classes)

        if self.n_classes_2 is not None:
            logits2 = self.fc3(self.dropout(x_reduced))
            return logits, logits2
        
        return logits

class DistilBERTMHGAttention(nn.Module):
    def __init__(self, encoding_dimension, PRE_TRAINED_MODEL_NAME, seq_len, n_classes, device, n_classes2 = None):
        super().__init__()

        if PRE_TRAINED_MODEL_NAME == "roberta-base":
            self.distil_bert_model = RobertaModel.from_pretrained(PRE_TRAINED_MODEL_NAME, output_hidden_states=True)
        else:
            self.distil_bert_model = AutoModel.from_pretrained(PRE_TRAINED_MODEL_NAME, output_hidden_states=True)

        self.encoding_dimension = encoding_dimension
        self.seq_len = seq_len
        self.n_classes = n_classes
        self.n_classes_2 = n_classes2
        self.device = device

        self.dropout = nn.Dropout(p=0.05)

        num_hidden_layers = self.distil_bert_model.config.num_hidden_layers

        # Una instancia de MultiHeadGAttention por capa
        self.multihead_gattentions = nn.ModuleList([
            GAttention(encoding_dimension, batch_first=True, device=device, seq_len=seq_len) for _ in range(num_hidden_layers)
        ])

        # Feed-forward final
        self.feed_forward = nn.Sequential(
            nn.Linear(encoding_dimension, encoding_dimension * 4),
            nn.ReLU(),
            nn.Linear(encoding_dimension * 4, encoding_dimension)
        )

        self.layer_norm_1 = nn.LayerNorm(encoding_dimension)
        self.layer_norm_2 = nn.LayerNorm(encoding_dimension)

        # Proyección para reducción de secuencia
        self.fc1 = nn.Linear(seq_len, 1)
        self.fc2 = nn.Linear(encoding_dimension, n_classes)
        if self.n_classes_2 is not None:
            self.fc3 = nn.Linear(encoding_dimension, self.n_classes_2)

    def forward(self, ids, mask, return_sigma=False, return_context=False):
        outputs = self.distil_bert_model(
            input_ids=ids,
            attention_mask=mask,
            return_dict=True,
            output_hidden_states=True,
            output_attentions=False
        )

        hidden_states = outputs.hidden_states  # List of (B, L, D)
        x = hidden_states[1]  # Saltamos la embedding layer
        all_layer_outputs = []

        for ga_layer, hidden in zip(self.multihead_gattentions, hidden_states[1:]):
            ga_out = ga_layer(hidden, mask, return_sigma, return_context)  # (B, L, D)
            all_layer_outputs.append(ga_out)


        # Agregación (promedio simple en este caso)
        stacked = torch.stack(all_layer_outputs, dim=0)  # (num_layers, B, L, D)
        ga_output = torch.mean(stacked, dim=0)  # (B, L, D)

        last_hidden_state = outputs.last_hidden_state  # (B, L, D)
        x = self.layer_norm_1(last_hidden_state + self.dropout(ga_output))

        ff_output = self.feed_forward(x)
        x = self.layer_norm_2(x + self.dropout(ff_output))  # (B, L, D)

        x_reduced = self.fc1(x.permute(0, 2, 1)).squeeze(-1)  # (B, D)

        logits = self.fc2(self.dropout(x_reduced))  # (B, n_classes)

        if self.n_classes_2 is not None:
            logits2 = self.fc3(self.dropout(x_reduced))
            return logits, logits2

        return logits

class BERTMHGAttention(nn.Module):
    def __init__(self, encoding_dimension, pretrained_model_name, seq_len, n_classes, device, n_classes2 = None):
        super().__init__()

        self.bert_model = AutoModel.from_pretrained(pretrained_model_name, output_hidden_states=True)

        self.encoding_dimension = encoding_dimension
        self.n_classes = n_classes
        self.n_classes_2 = n_classes2
        self.device = device
        self.seq_len = seq_len

        self.dropout = nn.Dropout(p=0.05)

        num_hidden_layers = self.bert_model.config.num_hidden_layers

        self.gattention_layers = nn.ModuleList([
            GAttention(encoding_dimension, batch_first=True, device=device, seq_len=seq_len)
            for _ in range(num_hidden_layers)
        ])

        self.feed_forward = nn.Sequential(
            nn.Linear(encoding_dimension, encoding_dimension * 4),
            nn.ReLU(),
            nn.Linear(encoding_dimension * 4, encoding_dimension)
        )

        self.layer_norm_1 = nn.LayerNorm(encoding_dimension)
        self.layer_norm_2 = nn.LayerNorm(encoding_dimension)

        self.fc1 = nn.Linear(seq_len, 1)
        self.fc2 = nn.Linear(encoding_dimension, n_classes)

        if self.n_classes_2 is not None:
            self.fc3 = nn.Linear(encoding_dimension, self.n_classes_2)

    def forward(self, ids, mask, token_type_ids, return_sigma=False, return_context=False):
        outputs = self.bert_model(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )

        hidden_states = outputs.hidden_states  # (num_layers + 1) x (B, L, D)
        x = hidden_states[1]  # skip embeddings (index 0)
        all_layer_outputs = []
        
        for ga_layer, hidden in zip(self.gattention_layers, hidden_states[1:]):
            ga_out = ga_layer(hidden, mask, return_sigma, return_context)  # (B, L, D)
            all_layer_outputs.append(ga_out)
        
        # Agregación (promedio simple en este caso)
        stacked = torch.stack(all_layer_outputs, dim=0)  # (num_layers, B, L, D)
        ga_output = torch.mean(stacked, dim=0)  # (B, L, D)

        last_hidden_state = outputs.last_hidden_state  # (B, L, D)
        x = self.layer_norm_1(last_hidden_state + self.dropout(ga_output))

        ff_output = self.feed_forward(x)
        x = self.layer_norm_2(x + self.dropout(ff_output))  # (B, L, D)

        x_reduced = self.fc1(x.permute(0, 2, 1)).squeeze(-1)  # (B, D)

        logits = self.fc2(self.dropout(x_reduced))  # (B, n_classes)

        if self.n_classes_2 is not None:
            logits2 = self.fc3(self.dropout(x_reduced))
            return logits, logits2
        
        return logits