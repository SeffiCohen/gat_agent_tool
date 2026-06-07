import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, TransformerConv, global_mean_pool, global_add_pool, global_max_pool, GlobalAttention
from torch_geometric.data import Data

# Constants
NODE_TYPE_OP = 0
NODE_TYPE_VAR = 1
POSITION_ROOT = 0
POSITION_LEFT = 1
POSITION_RIGHT = 2
NUM_POSITIONS = 3

# Edge type encoding for tree-aware graphs
EDGE_PARENT_TO_LEFT = 0
EDGE_PARENT_TO_RIGHT = 1
EDGE_LEFT_TO_PARENT = 2
EDGE_RIGHT_TO_PARENT = 3
EDGE_UNKNOWN = 4
NUM_EDGE_TYPES = 5

# Edge type encoding for tree-aware graphs
EDGE_PARENT_TO_LEFT = 0
EDGE_PARENT_TO_RIGHT = 1
EDGE_LEFT_TO_PARENT = 2
EDGE_RIGHT_TO_PARENT = 3
EDGE_UNKNOWN = 4
NUM_EDGE_TYPES = 5

# Activation function mapping
ACTIVATION_MAP = {"elu": nn.ELU, "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh, "gelu": nn.GELU}

class EfficientGAT(nn.Module):
    """
    Legacy EfficientGAT model.
    """
    def __init__(
        self,
        num_operators: int,
        num_features: int,
        embedding_dim_type: int,
        embedding_dim_id: int,
        hidden_channels_gat: int,
        out_channels_final: int,
        num_layers_gat: int = 4,
        heads_gat: int = 4,
        dropout_rate: float = 0.1,
        pooling_type_gat: str = "mean_add",
        activation_fn_gat = nn.ELU,
        embedding_dim_position: int = 16,
        num_constants: int = 0, # Ignored
    ):
        super().__init__()
        assert num_layers_gat >= 1, "Number of GAT layers must be at least 1."

        self.dropout_rate = dropout_rate
        self.pooling_type = pooling_type_gat

        # Embedding layers
        self.type_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim_type)
        self.op_id_embedding = nn.Embedding(num_embeddings=num_operators, embedding_dim=embedding_dim_id)
        self.var_id_embedding = nn.Embedding(num_embeddings=num_features, embedding_dim=embedding_dim_id)
        self.position_embedding = nn.Embedding(num_embeddings=NUM_POSITIONS, embedding_dim=embedding_dim_position)

        # Input dimension
        gat_conv_in_channels = embedding_dim_type + embedding_dim_id + embedding_dim_position

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        current_gat_dim = gat_conv_in_channels
        for i in range(num_layers_gat):
            self.convs.append(
                GATConv(
                    current_gat_dim if i == 0 else hidden_channels_gat,
                    hidden_channels_gat,
                    heads=heads_gat,
                    concat=False,
                    dropout=dropout_rate,
                    add_self_loops=True,
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels_gat))
            current_gat_dim = hidden_channels_gat

        pool_out_dim_mlp = hidden_channels_gat * (2 if pooling_type_gat == "mean_add" else 1)
        
        mlp_hidden_dim = pool_out_dim_mlp // 2
        if mlp_hidden_dim <= 0 and pool_out_dim_mlp > 0: mlp_hidden_dim = 1
        elif pool_out_dim_mlp == 0: mlp_hidden_dim = 1

        # Handle class or instance for activation
        if isinstance(activation_fn_gat, type):
            self.activation_module = activation_fn_gat()
        else:
            self.activation_module = activation_fn_gat

        self.mlp = nn.Sequential(
            nn.Linear(pool_out_dim_mlp, mlp_hidden_dim),
            self.activation_module,
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, out_channels_final),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x_ids, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x_ids.size(0), dtype=torch.long, device=x_ids.device)

        type_ids = x_ids[:, 0]
        specific_ids = x_ids[:, 1]
        
        if x_ids.size(1) >= 3:
            position_ids = x_ids[:, 2]
        else:
            position_ids = torch.zeros(x_ids.size(0), dtype=torch.long, device=x_ids.device)

        type_embeds = self.type_embedding(type_ids)
        position_embeds = self.position_embedding(position_ids)

        op_mask = (type_ids == NODE_TYPE_OP)
        var_mask = (type_ids == NODE_TYPE_VAR)

        specific_id_embeds = torch.zeros(x_ids.size(0), self.op_id_embedding.embedding_dim, device=x_ids.device)

        if op_mask.any():
            op_specific_ids = specific_ids[op_mask]
            # Safety clamp handled in wrapper/dataset usually, but good practice to have here if needed
            specific_id_embeds[op_mask] = self.op_id_embedding(op_specific_ids)
        
        if var_mask.any():
            var_specific_ids = specific_ids[var_mask]
            specific_id_embeds[var_mask] = self.var_id_embedding(var_specific_ids)
            
        x_embedded = torch.cat([type_embeds, specific_id_embeds, position_embeds], dim=1)
        
        x_conv = x_embedded
        res_connection = None 
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h_conv = conv(x_conv, edge_index)
            h_conv = bn(h_conv)
            h_conv = self.activation_module(h_conv)
            if res_connection is not None and i % 2 == 1 and h_conv.shape == res_connection.shape:
                h_conv = h_conv + res_connection  
            res_connection = h_conv
            x_conv = F.dropout(h_conv, p=self.dropout_rate, training=self.training)

        if self.pooling_type == "mean": g_pool = global_mean_pool(x_conv, batch)
        elif self.pooling_type == "add": g_pool = global_add_pool(x_conv, batch)
        elif self.pooling_type == "max": g_pool = global_max_pool(x_conv, batch)
        elif self.pooling_type == "mean_add":
            g_pool = torch.cat([global_mean_pool(x_conv, batch), global_add_pool(x_conv, batch)], dim=1)
        else: raise ValueError(f"Unknown pooling type {self.pooling_type}")

        return self.mlp(g_pool)


class AdvancedGAT(nn.Module):
    """
    Advanced GAT model using TransformerConv, LayerNorm, and GlobalAttention pooling.
    Designed for better ranking performance and stability.
    """
    def __init__(
        self,
        num_operators: int,
        num_features: int,
        embedding_dim_type: int,
        embedding_dim_id: int,
        hidden_channels_gat: int,
        out_channels_final: int,
        num_layers_gat: int = 6,
        heads_gat: int = 8,
        dropout_rate: float = 0.1,
        pooling_type_gat: str = "attention", # 'attention', 'mean', 'add', 'max', 'mean_add'
        activation_fn_gat = nn.GELU,
        embedding_dim_position: int = 16,
        num_constants: int = 0, # Ignored
    ):
        super().__init__()
        assert num_layers_gat >= 1, "Number of GAT layers must be at least 1."

        self.dropout_rate = dropout_rate
        self.pooling_type = pooling_type_gat

        # Embedding layers
        self.type_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim_type)
        self.op_id_embedding = nn.Embedding(num_embeddings=num_operators, embedding_dim=embedding_dim_id)
        self.var_id_embedding = nn.Embedding(num_embeddings=num_features, embedding_dim=embedding_dim_id)
        self.position_embedding = nn.Embedding(num_embeddings=NUM_POSITIONS, embedding_dim=embedding_dim_position)

        # Input dimension
        input_dim = embedding_dim_type + embedding_dim_id + embedding_dim_position

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # TransformerConv layers
        # We keep dimensions constant throughout layers
        # concat=True means output dim is heads * hidden_channels
        # We project or ensure dimensions match for residuals
        
        current_dim = input_dim
        self.heads = heads_gat
        self.hidden_channels = hidden_channels_gat
        
        for i in range(num_layers_gat):
            # TransformerConv
            # If concat=True, output is heads * out_channels
            # We want the output dimension to match 'hidden_channels_gat' * 'heads' (or similar stable dim)
            # But usually, we define hidden_channels as per-head dim or total dim.
            # PyG TransformerConv: out_channels is size of each head.
            # So total output size is out_channels * heads.
            
            # For the first layer, input is current_dim.
            # For subsequent layers, input is hidden_channels * heads (from previous layer).
            
            in_channels = current_dim
            out_channels = hidden_channels_gat // heads_gat if heads_gat > 0 else hidden_channels_gat
            
            # Ensure out_channels is at least 1
            out_channels = max(1, out_channels)
            
            self.convs.append(
                TransformerConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=heads_gat,
                    concat=True,
                    dropout=dropout_rate,
                    beta=True # Gating mechanism
                )
            )
            
            # Output dim of this layer
            next_dim = out_channels * heads_gat
            
            # LayerNorm
            self.norms.append(nn.LayerNorm(next_dim))
            
            current_dim = next_dim

        # Final node representation dimension
        node_dim = current_dim

        # Activation
        if isinstance(activation_fn_gat, type):
            self.activation_module = activation_fn_gat()
        else:
            self.activation_module = activation_fn_gat

        # Pooling mechanism
        if pooling_type_gat == "attention":
            # GlobalAttention with a learnable gating function
            self.pool = GlobalAttention(
                gate_nn=nn.Sequential(
                    nn.Linear(node_dim, node_dim // 2),
                    nn.ReLU(),
                    nn.Linear(node_dim // 2, 1)
                )
            )
            pool_out_dim = node_dim
        elif pooling_type_gat == "mean":
            self.pool = global_mean_pool
            pool_out_dim = node_dim
        elif pooling_type_gat == "add":
            self.pool = global_add_pool
            pool_out_dim = node_dim
        elif pooling_type_gat == "max":
            self.pool = global_max_pool
            pool_out_dim = node_dim
        elif pooling_type_gat == "mean_add":
            self.pool = lambda x, batch: torch.cat([global_mean_pool(x, batch), global_add_pool(x, batch)], dim=1)
            pool_out_dim = node_dim * 2
        else:
            raise ValueError(f"Unknown pooling type {pooling_type_gat}")

        # Regression Head
        mlp_hidden_dim = pool_out_dim // 2
        self.mlp = nn.Sequential(
            nn.Linear(pool_out_dim, mlp_hidden_dim),
            self.activation_module,
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            self.activation_module,
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim // 2, out_channels_final)
        )

    def forward(self, data: Data) -> torch.Tensor:
        x_ids, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x_ids.size(0), dtype=torch.long, device=x_ids.device)

        type_ids = x_ids[:, 0]
        specific_ids = x_ids[:, 1]
        
        if x_ids.size(1) >= 3:
            position_ids = x_ids[:, 2]
        else:
            position_ids = torch.zeros(x_ids.size(0), dtype=torch.long, device=x_ids.device)

        # Embeddings
        type_embeds = self.type_embedding(type_ids)
        position_embeds = self.position_embedding(position_ids)

        op_mask = (type_ids == NODE_TYPE_OP)
        var_mask = (type_ids == NODE_TYPE_VAR)

        specific_id_embeds = torch.zeros(x_ids.size(0), self.op_id_embedding.embedding_dim, device=x_ids.device)

        if op_mask.any():
            op_specific_ids = specific_ids[op_mask]
            # Clamp for safety
            op_specific_ids = torch.clamp(op_specific_ids, 0, self.op_id_embedding.num_embeddings - 1)
            specific_id_embeds[op_mask] = self.op_id_embedding(op_specific_ids)
        
        if var_mask.any():
            var_specific_ids = specific_ids[var_mask]
            # Clamp for safety
            var_specific_ids = torch.clamp(var_specific_ids, 0, self.var_id_embedding.num_embeddings - 1)
            specific_id_embeds[var_mask] = self.var_id_embedding(var_specific_ids)
            
        x = torch.cat([type_embeds, specific_id_embeds, position_embeds], dim=1)
        
        # Transformer Blocks
        for conv, norm in zip(self.convs, self.norms):
            x_in = x
            x = conv(x, edge_index)
            x = norm(x)
            x = self.activation_module(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
            # Residual connection if dimensions match
            if x.shape == x_in.shape:
                x = x + x_in

        # Pooling
        if self.pooling_type == "attention":
            g_pool = self.pool(x, batch)
        elif self.pooling_type == "mean_add":
            g_pool = self.pool(x, batch) # lambda defined in init
        else:
            g_pool = self.pool(x, batch) # functional pooling

        # Regression Head
        out = self.mlp(g_pool)
        return out


class TreeAwareGAT(nn.Module):
    """
    Tree-aware GAT with optional edge-type and depth embeddings.
    Falls back to v1 behavior if depth/edge_attr are missing.
    """
    def __init__(
        self,
        num_operators: int,
        num_features: int,
        embedding_dim_type: int,
        embedding_dim_id: int,
        hidden_channels_gat: int,
        out_channels_final: int,
        num_layers_gat: int = 6,
        heads_gat: int = 8,
        dropout_rate: float = 0.1,
        pooling_type_gat: str = "attention", # 'attention', 'root', 'root+attention', 'mean', 'add', 'max', 'mean_add'
        activation_fn_gat = nn.GELU,
        embedding_dim_position: int = 16,
        embedding_dim_depth: int = 16,
        edge_embedding_dim: int = 8,
        num_edge_types: int = NUM_EDGE_TYPES,
        max_depth: int = 32,
        num_constants: int = 0, # Ignored
    ):
        super().__init__()
        assert num_layers_gat >= 1, "Number of GAT layers must be at least 1."
        self.dropout_rate = dropout_rate
        self.pooling_type = pooling_type_gat
        self.num_edge_types = max(1, int(num_edge_types))
        self.max_depth = max(1, int(max_depth))
        self.edge_embedding_dim = int(edge_embedding_dim)

        # Embedding layers
        self.type_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim_type)
        self.op_id_embedding = nn.Embedding(num_embeddings=num_operators, embedding_dim=embedding_dim_id)
        self.var_id_embedding = nn.Embedding(num_embeddings=num_features, embedding_dim=embedding_dim_id)
        self.position_embedding = nn.Embedding(num_embeddings=NUM_POSITIONS, embedding_dim=embedding_dim_position)
        self.depth_embedding = nn.Embedding(num_embeddings=self.max_depth + 1, embedding_dim=embedding_dim_depth)
        self.edge_type_embedding = nn.Embedding(num_embeddings=self.num_edge_types, embedding_dim=self.edge_embedding_dim)

        # Input dimension
        input_dim = embedding_dim_type + embedding_dim_id + embedding_dim_position + embedding_dim_depth

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        current_dim = input_dim
        self.heads = heads_gat
        self.hidden_channels = hidden_channels_gat

        for i in range(num_layers_gat):
            in_channels = current_dim
            out_channels = hidden_channels_gat // heads_gat if heads_gat > 0 else hidden_channels_gat
            out_channels = max(1, out_channels)
            self.convs.append(
                TransformerConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=heads_gat,
                    concat=True,
                    dropout=dropout_rate,
                    beta=True,
                    edge_dim=self.edge_embedding_dim,
                )
            )
            next_dim = out_channels * heads_gat
            self.norms.append(nn.LayerNorm(next_dim))
            current_dim = next_dim

        node_dim = current_dim

        if isinstance(activation_fn_gat, type):
            self.activation_module = activation_fn_gat()
        else:
            self.activation_module = activation_fn_gat

        if pooling_type_gat in {"attention", "root+attention"}:
            self.attn_pool = GlobalAttention(
                gate_nn=nn.Sequential(
                    nn.Linear(node_dim, node_dim // 2),
                    nn.ReLU(),
                    nn.Linear(node_dim // 2, 1)
                )
            )
        else:
            self.attn_pool = None

        if pooling_type_gat == "attention":
            self.pool = self.attn_pool
            pool_out_dim = node_dim
        elif pooling_type_gat == "root":
            self.pool = None
            pool_out_dim = node_dim
        elif pooling_type_gat == "root+attention":
            self.pool = self.attn_pool
            pool_out_dim = node_dim * 2
        elif pooling_type_gat == "mean":
            self.pool = global_mean_pool
            pool_out_dim = node_dim
        elif pooling_type_gat == "add":
            self.pool = global_add_pool
            pool_out_dim = node_dim
        elif pooling_type_gat == "max":
            self.pool = global_max_pool
            pool_out_dim = node_dim
        elif pooling_type_gat == "mean_add":
            self.pool = lambda x, batch: torch.cat([global_mean_pool(x, batch), global_add_pool(x, batch)], dim=1)
            pool_out_dim = node_dim * 2
        else:
            raise ValueError(f"Unknown pooling type {pooling_type_gat}")

        mlp_hidden_dim = pool_out_dim // 2
        self.mlp = nn.Sequential(
            nn.Linear(pool_out_dim, mlp_hidden_dim),
            self.activation_module,
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            self.activation_module,
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim // 2, out_channels_final)
        )

    def _get_root_indices(self, data: Data) -> torch.Tensor:
        if hasattr(data, "ptr") and data.ptr is not None:
            return data.ptr[:-1]
        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            root_idx = torch.zeros(num_graphs, dtype=torch.long, device=batch.device)
            for g in range(num_graphs):
                root_idx[g] = (batch == g).nonzero(as_tuple=False)[0, 0]
            return root_idx
        return torch.tensor([0], dtype=torch.long, device=data.x.device)

    def forward(self, data: Data) -> torch.Tensor:
        x_ids, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x_ids.size(0), dtype=torch.long, device=x_ids.device)

        type_ids = x_ids[:, 0]
        specific_ids = x_ids[:, 1]
        position_ids = x_ids[:, 2] if x_ids.size(1) >= 3 else torch.zeros(x_ids.size(0), dtype=torch.long, device=x_ids.device)
        depth_ids = x_ids[:, 3] if x_ids.size(1) >= 4 else torch.zeros(x_ids.size(0), dtype=torch.long, device=x_ids.device)

        # Embeddings
        type_embeds = self.type_embedding(type_ids)
        position_embeds = self.position_embedding(position_ids)
        depth_ids = torch.clamp(depth_ids, 0, self.depth_embedding.num_embeddings - 1)
        depth_embeds = self.depth_embedding(depth_ids)

        op_mask = (type_ids == NODE_TYPE_OP)
        var_mask = (type_ids == NODE_TYPE_VAR)

        specific_id_embeds = torch.zeros(x_ids.size(0), self.op_id_embedding.embedding_dim, device=x_ids.device)
        if op_mask.any():
            op_specific_ids = torch.clamp(specific_ids[op_mask], 0, self.op_id_embedding.num_embeddings - 1)
            specific_id_embeds[op_mask] = self.op_id_embedding(op_specific_ids)
        if var_mask.any():
            var_specific_ids = torch.clamp(specific_ids[var_mask], 0, self.var_id_embedding.num_embeddings - 1)
            specific_id_embeds[var_mask] = self.var_id_embedding(var_specific_ids)

        x = torch.cat([type_embeds, specific_id_embeds, position_embeds, depth_embeds], dim=1)

        edge_attr = getattr(data, "edge_attr", None)
        edge_attr_embeds = None
        if edge_attr is not None and edge_attr.numel() > 0:
            if edge_attr.dim() > 1:
                edge_attr = edge_attr.view(-1)
            edge_attr = torch.clamp(edge_attr.long(), 0, self.edge_type_embedding.num_embeddings - 1)
            edge_attr_embeds = self.edge_type_embedding(edge_attr)

        # Transformer Blocks
        for conv, norm in zip(self.convs, self.norms):
            x_in = x
            if edge_attr_embeds is not None:
                x = conv(x, edge_index, edge_attr_embeds)
            else:
                x = conv(x, edge_index)
            x = norm(x)
            x = self.activation_module(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            if x.shape == x_in.shape:
                x = x + x_in

        # Pooling
        if self.pooling_type == "root":
            root_idx = self._get_root_indices(data)
            g_pool = x[root_idx]
        elif self.pooling_type == "root+attention":
            root_idx = self._get_root_indices(data)
            attn_pool = self.pool(x, batch) if self.pool is not None else global_mean_pool(x, batch)
            g_pool = torch.cat([x[root_idx], attn_pool], dim=1)
        elif self.pooling_type == "attention":
            g_pool = self.pool(x, batch)
        elif self.pooling_type == "mean_add":
            g_pool = self.pool(x, batch)
        else:
            g_pool = self.pool(x, batch)

        out = self.mlp(g_pool)
        return out

# Activation function mapping
ACTIVATION_MAP = {"elu": nn.ELU, "relu": nn.ReLU, "leaky_relu": nn.LeakyReLU, "tanh": nn.Tanh, "gelu": nn.GELU}

class EfficientGAT(nn.Module):
    """
    Legacy EfficientGAT model.
    """
    def __init__(
        self,
        num_operators: int,
        num_features: int,
        embedding_dim_type: int,
        embedding_dim_id: int,
        hidden_channels_gat: int,
        out_channels_final: int,
        num_layers_gat: int = 4,
        heads_gat: int = 4,
        dropout_rate: float = 0.1,
        pooling_type_gat: str = "mean_add",
        activation_fn_gat = nn.ELU,
        embedding_dim_position: int = 16,
        num_constants: int = 0, # Ignored
    ):
        super().__init__()
        assert num_layers_gat >= 1, "Number of GAT layers must be at least 1."

        self.dropout_rate = dropout_rate
        self.pooling_type = pooling_type_gat

        # Embedding layers
        self.type_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim_type)
        self.op_id_embedding = nn.Embedding(num_embeddings=num_operators, embedding_dim=embedding_dim_id)
        self.var_id_embedding = nn.Embedding(num_embeddings=num_features, embedding_dim=embedding_dim_id)
        self.position_embedding = nn.Embedding(num_embeddings=NUM_POSITIONS, embedding_dim=embedding_dim_position)

        # Input dimension
        gat_conv_in_channels = embedding_dim_type + embedding_dim_id + embedding_dim_position

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        current_gat_dim = gat_conv_in_channels
        for i in range(num_layers_gat):
            self.convs.append(
                GATConv(
                    current_gat_dim if i == 0 else hidden_channels_gat,
                    hidden_channels_gat,
                    heads=heads_gat,
                    concat=False,
                    dropout=dropout_rate,
                    add_self_loops=True,
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_channels_gat))
            current_gat_dim = hidden_channels_gat

        pool_out_dim_mlp = hidden_channels_gat * (2 if pooling_type_gat == "mean_add" else 1)
        
        mlp_hidden_dim = pool_out_dim_mlp // 2
        if mlp_hidden_dim <= 0 and pool_out_dim_mlp > 0: mlp_hidden_dim = 1
        elif pool_out_dim_mlp == 0: mlp_hidden_dim = 1

        # Handle class or instance for activation
        if isinstance(activation_fn_gat, type):
            self.activation_module = activation_fn_gat()
        else:
            self.activation_module = activation_fn_gat

        self.mlp = nn.Sequential(
            nn.Linear(pool_out_dim_mlp, mlp_hidden_dim),
            self.activation_module,
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, out_channels_final),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x_ids, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x_ids.size(0), dtype=torch.long, device=x_ids.device)

        type_ids = x_ids[:, 0]
        specific_ids = x_ids[:, 1]
        
        if x_ids.size(1) >= 3:
            position_ids = x_ids[:, 2]
        else:
            position_ids = torch.zeros(x_ids.size(0), dtype=torch.long, device=x_ids.device)

        type_embeds = self.type_embedding(type_ids)
        position_embeds = self.position_embedding(position_ids)

        op_mask = (type_ids == NODE_TYPE_OP)
        var_mask = (type_ids == NODE_TYPE_VAR)

        specific_id_embeds = torch.zeros(x_ids.size(0), self.op_id_embedding.embedding_dim, device=x_ids.device)

        if op_mask.any():
            op_specific_ids = specific_ids[op_mask]
            # Safety clamp handled in wrapper/dataset usually, but good practice to have here if needed
            specific_id_embeds[op_mask] = self.op_id_embedding(op_specific_ids)
        
        if var_mask.any():
            var_specific_ids = specific_ids[var_mask]
            specific_id_embeds[var_mask] = self.var_id_embedding(var_specific_ids)
            
        x_embedded = torch.cat([type_embeds, specific_id_embeds, position_embeds], dim=1)
        
        x_conv = x_embedded
        res_connection = None 
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            h_conv = conv(x_conv, edge_index)
            h_conv = bn(h_conv)
            h_conv = self.activation_module(h_conv)
            if res_connection is not None and i % 2 == 1 and h_conv.shape == res_connection.shape:
                h_conv = h_conv + res_connection  
            res_connection = h_conv
            x_conv = F.dropout(h_conv, p=self.dropout_rate, training=self.training)

        if self.pooling_type == "mean": g_pool = global_mean_pool(x_conv, batch)
        elif self.pooling_type == "add": g_pool = global_add_pool(x_conv, batch)
        elif self.pooling_type == "max": g_pool = global_max_pool(x_conv, batch)
        elif self.pooling_type == "mean_add":
            g_pool = torch.cat([global_mean_pool(x_conv, batch), global_add_pool(x_conv, batch)], dim=1)
        else: raise ValueError(f"Unknown pooling type {self.pooling_type}")

        return self.mlp(g_pool)


class AdvancedGAT(nn.Module):
    """
    Advanced GAT model using TransformerConv, LayerNorm, and GlobalAttention pooling.
    Designed for better ranking performance and stability.
    """
    def __init__(
        self,
        num_operators: int,
        num_features: int,
        embedding_dim_type: int,
        embedding_dim_id: int,
        hidden_channels_gat: int,
        out_channels_final: int,
        num_layers_gat: int = 6,
        heads_gat: int = 8,
        dropout_rate: float = 0.1,
        pooling_type_gat: str = "attention", # 'attention', 'mean', 'add', 'max', 'mean_add'
        activation_fn_gat = nn.GELU,
        embedding_dim_position: int = 16,
        num_constants: int = 0, # Ignored
    ):
        super().__init__()
        assert num_layers_gat >= 1, "Number of GAT layers must be at least 1."

        self.dropout_rate = dropout_rate
        self.pooling_type = pooling_type_gat

        # Embedding layers
        self.type_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim_type)
        self.op_id_embedding = nn.Embedding(num_embeddings=num_operators, embedding_dim=embedding_dim_id)
        self.var_id_embedding = nn.Embedding(num_embeddings=num_features, embedding_dim=embedding_dim_id)
        self.position_embedding = nn.Embedding(num_embeddings=NUM_POSITIONS, embedding_dim=embedding_dim_position)

        # Input dimension
        input_dim = embedding_dim_type + embedding_dim_id + embedding_dim_position

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # TransformerConv layers
        # We keep dimensions constant throughout layers
        # concat=True means output dim is heads * hidden_channels
        # We project or ensure dimensions match for residuals
        
        current_dim = input_dim
        self.heads = heads_gat
        self.hidden_channels = hidden_channels_gat
        
        for i in range(num_layers_gat):
            # TransformerConv
            # If concat=True, output is heads * out_channels
            # We want the output dimension to match 'hidden_channels_gat' * 'heads' (or similar stable dim)
            # But usually, we define hidden_channels as per-head dim or total dim.
            # PyG TransformerConv: out_channels is size of each head.
            # So total output size is out_channels * heads.
            
            # For the first layer, input is current_dim.
            # For subsequent layers, input is hidden_channels * heads (from previous layer).
            
            in_channels = current_dim
            out_channels = hidden_channels_gat // heads_gat if heads_gat > 0 else hidden_channels_gat
            
            # Ensure out_channels is at least 1
            out_channels = max(1, out_channels)
            
            self.convs.append(
                TransformerConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=heads_gat,
                    concat=True,
                    dropout=dropout_rate,
                    beta=True # Gating mechanism
                )
            )
            
            # Output dim of this layer
            next_dim = out_channels * heads_gat
            
            # LayerNorm
            self.norms.append(nn.LayerNorm(next_dim))
            
            current_dim = next_dim

        # Final node representation dimension
        node_dim = current_dim

        # Activation
        if isinstance(activation_fn_gat, type):
            self.activation_module = activation_fn_gat()
        else:
            self.activation_module = activation_fn_gat

        # Pooling mechanism
        if pooling_type_gat == "attention":
            # GlobalAttention with a learnable gating function
            self.pool = GlobalAttention(
                gate_nn=nn.Sequential(
                    nn.Linear(node_dim, node_dim // 2),
                    nn.ReLU(),
                    nn.Linear(node_dim // 2, 1)
                )
            )
            pool_out_dim = node_dim
        elif pooling_type_gat == "mean":
            self.pool = global_mean_pool
            pool_out_dim = node_dim
        elif pooling_type_gat == "add":
            self.pool = global_add_pool
            pool_out_dim = node_dim
        elif pooling_type_gat == "max":
            self.pool = global_max_pool
            pool_out_dim = node_dim
        elif pooling_type_gat == "mean_add":
            self.pool = lambda x, batch: torch.cat([global_mean_pool(x, batch), global_add_pool(x, batch)], dim=1)
            pool_out_dim = node_dim * 2
        else:
            raise ValueError(f"Unknown pooling type {pooling_type_gat}")

        # Regression Head
        mlp_hidden_dim = pool_out_dim // 2
        self.mlp = nn.Sequential(
            nn.Linear(pool_out_dim, mlp_hidden_dim),
            self.activation_module,
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            self.activation_module,
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim // 2, out_channels_final)
        )

    def forward(self, data: Data) -> torch.Tensor:
        x_ids, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x_ids.size(0), dtype=torch.long, device=x_ids.device)

        type_ids = x_ids[:, 0]
        specific_ids = x_ids[:, 1]
        
        if x_ids.size(1) >= 3:
            position_ids = x_ids[:, 2]
        else:
            position_ids = torch.zeros(x_ids.size(0), dtype=torch.long, device=x_ids.device)

        # Embeddings
        type_embeds = self.type_embedding(type_ids)
        position_embeds = self.position_embedding(position_ids)

        op_mask = (type_ids == NODE_TYPE_OP)
        var_mask = (type_ids == NODE_TYPE_VAR)

        specific_id_embeds = torch.zeros(x_ids.size(0), self.op_id_embedding.embedding_dim, device=x_ids.device)

        if op_mask.any():
            op_specific_ids = specific_ids[op_mask]
            # Clamp for safety
            op_specific_ids = torch.clamp(op_specific_ids, 0, self.op_id_embedding.num_embeddings - 1)
            specific_id_embeds[op_mask] = self.op_id_embedding(op_specific_ids)
        
        if var_mask.any():
            var_specific_ids = specific_ids[var_mask]
            # Clamp for safety
            var_specific_ids = torch.clamp(var_specific_ids, 0, self.var_id_embedding.num_embeddings - 1)
            specific_id_embeds[var_mask] = self.var_id_embedding(var_specific_ids)
            
        x = torch.cat([type_embeds, specific_id_embeds, position_embeds], dim=1)
        
        # Transformer Blocks
        for conv, norm in zip(self.convs, self.norms):
            x_in = x
            x = conv(x, edge_index)
            x = norm(x)
            x = self.activation_module(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
            # Residual connection if dimensions match
            if x.shape == x_in.shape:
                x = x + x_in

        # Pooling
        if self.pooling_type == "attention":
            g_pool = self.pool(x, batch)
        elif self.pooling_type == "mean_add":
            g_pool = self.pool(x, batch) # lambda defined in init
        else:
            g_pool = self.pool(x, batch) # functional pooling

        # Regression Head
        out = self.mlp(g_pool)
        return out


class TreeAwareGAT(nn.Module):
    """
    Tree-aware GAT with optional edge-type and depth embeddings.
    Falls back to v1 behavior if depth/edge_attr are missing.
    """
    def __init__(
        self,
        num_operators: int,
        num_features: int,
        embedding_dim_type: int,
        embedding_dim_id: int,
        hidden_channels_gat: int,
        out_channels_final: int,
        num_layers_gat: int = 6,
        heads_gat: int = 8,
        dropout_rate: float = 0.1,
        pooling_type_gat: str = "attention", # 'attention', 'root', 'root+attention', 'mean', 'add', 'max', 'mean_add'
        activation_fn_gat = nn.GELU,
        embedding_dim_position: int = 16,
        embedding_dim_depth: int = 16,
        edge_embedding_dim: int = 8,
        num_edge_types: int = NUM_EDGE_TYPES,
        max_depth: int = 32,
        num_constants: int = 0, # Ignored
    ):
        super().__init__()
        assert num_layers_gat >= 1, "Number of GAT layers must be at least 1."
        self.dropout_rate = dropout_rate
        self.pooling_type = pooling_type_gat
        self.num_edge_types = max(1, int(num_edge_types))
        self.max_depth = max(1, int(max_depth))
        self.edge_embedding_dim = int(edge_embedding_dim)

        # Embedding layers
        self.type_embedding = nn.Embedding(num_embeddings=2, embedding_dim=embedding_dim_type)
        self.op_id_embedding = nn.Embedding(num_embeddings=num_operators, embedding_dim=embedding_dim_id)
        self.var_id_embedding = nn.Embedding(num_embeddings=num_features, embedding_dim=embedding_dim_id)
        self.position_embedding = nn.Embedding(num_embeddings=NUM_POSITIONS, embedding_dim=embedding_dim_position)
        self.depth_embedding = nn.Embedding(num_embeddings=self.max_depth + 1, embedding_dim=embedding_dim_depth)
        self.edge_type_embedding = nn.Embedding(num_embeddings=self.num_edge_types, embedding_dim=self.edge_embedding_dim)

        # Input dimension
        input_dim = embedding_dim_type + embedding_dim_id + embedding_dim_position + embedding_dim_depth

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        current_dim = input_dim
        self.heads = heads_gat
        self.hidden_channels = hidden_channels_gat

        for i in range(num_layers_gat):
            in_channels = current_dim
            out_channels = hidden_channels_gat // heads_gat if heads_gat > 0 else hidden_channels_gat
            out_channels = max(1, out_channels)
            self.convs.append(
                TransformerConv(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    heads=heads_gat,
                    concat=True,
                    dropout=dropout_rate,
                    beta=True,
                    edge_dim=self.edge_embedding_dim,
                )
            )
            next_dim = out_channels * heads_gat
            self.norms.append(nn.LayerNorm(next_dim))
            current_dim = next_dim

        node_dim = current_dim

        if isinstance(activation_fn_gat, type):
            self.activation_module = activation_fn_gat()
        else:
            self.activation_module = activation_fn_gat

        if pooling_type_gat in {"attention", "root+attention"}:
            self.attn_pool = GlobalAttention(
                gate_nn=nn.Sequential(
                    nn.Linear(node_dim, node_dim // 2),
                    nn.ReLU(),
                    nn.Linear(node_dim // 2, 1)
                )
            )
        else:
            self.attn_pool = None

        if pooling_type_gat == "attention":
            self.pool = self.attn_pool
            pool_out_dim = node_dim
        elif pooling_type_gat == "root":
            self.pool = None
            pool_out_dim = node_dim
        elif pooling_type_gat == "root+attention":
            self.pool = self.attn_pool
            pool_out_dim = node_dim * 2
        elif pooling_type_gat == "mean":
            self.pool = global_mean_pool
            pool_out_dim = node_dim
        elif pooling_type_gat == "add":
            self.pool = global_add_pool
            pool_out_dim = node_dim
        elif pooling_type_gat == "max":
            self.pool = global_max_pool
            pool_out_dim = node_dim
        elif pooling_type_gat == "mean_add":
            self.pool = lambda x, batch: torch.cat([global_mean_pool(x, batch), global_add_pool(x, batch)], dim=1)
            pool_out_dim = node_dim * 2
        else:
            raise ValueError(f"Unknown pooling type {pooling_type_gat}")

        mlp_hidden_dim = pool_out_dim // 2
        self.mlp = nn.Sequential(
            nn.Linear(pool_out_dim, mlp_hidden_dim),
            self.activation_module,
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim // 2),
            self.activation_module,
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_dim // 2, out_channels_final)
        )

    def _get_root_indices(self, data: Data) -> torch.Tensor:
        if hasattr(data, "ptr") and data.ptr is not None:
            return data.ptr[:-1]
        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            root_idx = torch.zeros(num_graphs, dtype=torch.long, device=batch.device)
            for g in range(num_graphs):
                root_idx[g] = (batch == g).nonzero(as_tuple=False)[0, 0]
            return root_idx
        return torch.tensor([0], dtype=torch.long, device=data.x.device)

    def forward(self, data: Data) -> torch.Tensor:
        x_ids, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x_ids.size(0), dtype=torch.long, device=x_ids.device)

        type_ids = x_ids[:, 0]
        specific_ids = x_ids[:, 1]
        position_ids = x_ids[:, 2] if x_ids.size(1) >= 3 else torch.zeros(x_ids.size(0), dtype=torch.long, device=x_ids.device)
        depth_ids = x_ids[:, 3] if x_ids.size(1) >= 4 else torch.zeros(x_ids.size(0), dtype=torch.long, device=x_ids.device)

        # Embeddings
        type_embeds = self.type_embedding(type_ids)
        position_embeds = self.position_embedding(position_ids)
        depth_ids = torch.clamp(depth_ids, 0, self.depth_embedding.num_embeddings - 1)
        depth_embeds = self.depth_embedding(depth_ids)

        op_mask = (type_ids == NODE_TYPE_OP)
        var_mask = (type_ids == NODE_TYPE_VAR)

        specific_id_embeds = torch.zeros(x_ids.size(0), self.op_id_embedding.embedding_dim, device=x_ids.device)
        if op_mask.any():
            op_specific_ids = torch.clamp(specific_ids[op_mask], 0, self.op_id_embedding.num_embeddings - 1)
            specific_id_embeds[op_mask] = self.op_id_embedding(op_specific_ids)
        if var_mask.any():
            var_specific_ids = torch.clamp(specific_ids[var_mask], 0, self.var_id_embedding.num_embeddings - 1)
            specific_id_embeds[var_mask] = self.var_id_embedding(var_specific_ids)

        x = torch.cat([type_embeds, specific_id_embeds, position_embeds, depth_embeds], dim=1)

        edge_attr = getattr(data, "edge_attr", None)
        edge_attr_embeds = None
        if edge_attr is not None and edge_attr.numel() > 0:
            if edge_attr.dim() > 1:
                edge_attr = edge_attr.view(-1)
            edge_attr = torch.clamp(edge_attr.long(), 0, self.edge_type_embedding.num_embeddings - 1)
            edge_attr_embeds = self.edge_type_embedding(edge_attr)

        # Transformer Blocks
        for conv, norm in zip(self.convs, self.norms):
            x_in = x
            if edge_attr_embeds is not None:
                x = conv(x, edge_index, edge_attr_embeds)
            else:
                x = conv(x, edge_index)
            x = norm(x)
            x = self.activation_module(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            if x.shape == x_in.shape:
                x = x + x_in

        # Pooling
        if self.pooling_type == "root":
            root_idx = self._get_root_indices(data)
            g_pool = x[root_idx]
        elif self.pooling_type == "root+attention":
            root_idx = self._get_root_indices(data)
            attn_pool = self.pool(x, batch) if self.pool is not None else global_mean_pool(x, batch)
            g_pool = torch.cat([x[root_idx], attn_pool], dim=1)
        elif self.pooling_type == "attention":
            g_pool = self.pool(x, batch)
        elif self.pooling_type == "mean_add":
            g_pool = self.pool(x, batch)
        else:
            g_pool = self.pool(x, batch)

        out = self.mlp(g_pool)
        return out
