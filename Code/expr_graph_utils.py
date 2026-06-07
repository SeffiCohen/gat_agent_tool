#!/usr/bin/env python3
"""
Expression Graph Utilities
==========================

Utilities for converting between expression strings and graph representations.
Consistent with Xtree_Gen_714_train_NB_SHM.py

ALIGNMENT: Maintains consistency with all other modules
- Node types: NODE_TYPE_OP=0, NODE_TYPE_VAR=1 (constants removed for simplicity)
- Operators: ["+", "-", "*", "/"]
- Expressions use only features and arithmetic operations (no numeric constants)
"""

import os
import re
import logging
from typing import Dict, Optional, Tuple, List, Union
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

# Constants (consistent across all files)
# Only 2 node types: operator and variable (constants removed)
NODE_TYPE_OP = 0
NODE_TYPE_VAR = 1
OPERATORS = ["+", "-", "*", "/"]

# Legacy constant support (deprecated - kept for backward compatibility)
# Constants are no longer used in the current architecture
SUPPORTED_CONSTANTS = []  # No constants supported
CONSTANT_MAPPING = {}     # Empty mapping for backward compatibility

# Position encoding constants for child nodes
# This enables the model to distinguish between left and right operands
# which is critical for non-commutative operations like subtraction and division
POSITION_ROOT = 0      # Root node or operator
POSITION_LEFT = 1      # Left child operand
POSITION_RIGHT = 2     # Right child operand

# Edge type encoding for tree-aware graphs
EDGE_PARENT_TO_LEFT = 0
EDGE_PARENT_TO_RIGHT = 1
EDGE_LEFT_TO_PARENT = 2
EDGE_RIGHT_TO_PARENT = 3
EDGE_UNKNOWN = 4
NUM_EDGE_TYPES = 5


class ExpressionParser:
    """Parse expression strings into tree structures"""
    
    def __init__(self):
        self.operators = set(OPERATORS)
        self.precedence = {"+": 1, "-": 1, "*": 2, "/": 2}
    
    def tokenize(self, expr: str) -> List[str]:
        """Tokenize expression string"""
        # Remove extra spaces and handle parentheses
        expr = expr.strip()
        
        # Split by operators and parentheses while keeping them
        pattern = r'([\+\-\*/\(\)])'
        tokens = re.split(pattern, expr)
        
        # Filter out empty tokens and strip whitespace
        tokens = [t.strip() for t in tokens if t.strip()]
        
        return tokens
    
    def parse(self, expr: str) -> Optional['ParseTreeNode']:
        """Parse expression string into a tree"""
        tokens = self.tokenize(expr)
        if not tokens:
            return None
        
        try:
            result, _ = self._parse_expression(tokens, 0)
            return result
        except Exception as e:
            logging.debug(f"Failed to parse expression '{expr}': {e}")
            return None
    
    def _parse_expression(self, tokens: List[str], pos: int) -> Tuple[Optional['ParseTreeNode'], int]:
        """Parse expression with precedence"""
        left, pos = self._parse_term(tokens, pos)
        if left is None:
            return None, pos
        
        while pos < len(tokens) and tokens[pos] in ["+", "-"]:
            op = tokens[pos]
            pos += 1
            right, pos = self._parse_term(tokens, pos)
            if right is None:
                return None, pos
            left = ParseTreeNode(op, left, right)
        
        return left, pos
    
    def _parse_term(self, tokens: List[str], pos: int) -> Tuple[Optional['ParseTreeNode'], int]:
        """Parse term (higher precedence)"""
        left, pos = self._parse_factor(tokens, pos)
        if left is None:
            return None, pos
        
        while pos < len(tokens) and tokens[pos] in ["*", "/"]:
            op = tokens[pos]
            pos += 1
            right, pos = self._parse_factor(tokens, pos)
            if right is None:
                return None, pos
            left = ParseTreeNode(op, left, right)
        
        return left, pos
    
    def _parse_factor(self, tokens: List[str], pos: int) -> Tuple[Optional['ParseTreeNode'], int]:
        """Parse factor (parentheses or variable - numeric constants are rejected)"""
        if pos >= len(tokens):
            return None, pos
        
        token = tokens[pos]
        
        if token == "(":
            pos += 1
            expr, pos = self._parse_expression(tokens, pos)
            if pos < len(tokens) and tokens[pos] == ")":
                pos += 1
                return expr, pos
            else:
                return None, pos
        elif token not in self.operators and token not in ["(", ")"]:
            # Check if it's a numeric constant - reject if so
            try:
                float(token)
                # Numeric constant detected - reject the expression
                logging.debug(f"Numeric constant '{token}' not supported, rejecting expression")
                return None, pos
            except ValueError:
                # Variable/feature name - this is what we want
                node = ParseTreeNode(None, token)
                node.is_constant = False
                return node, pos + 1
        else:
            return None, pos


class ParseTreeNode:
    """Simple tree node for parsed expressions"""
    
    def __init__(self, op: Optional[str], left=None, right=None):
        self.op = op
        self.left = left
        self.right = right
        self.value = left if op is None else None  # For leaf nodes
        self.is_constant = False  # Whether this is a numeric constant
        self.numeric_value = None  # The actual numeric value if constant
    
    def is_leaf(self) -> bool:
        return self.op is None
    
    def to_string(self) -> str:
        if self.is_leaf():
            return str(self.value)
        else:
            left_str = self.left.to_string() if isinstance(self.left, ParseTreeNode) else str(self.left)
            right_str = self.right.to_string() if isinstance(self.right, ParseTreeNode) else str(self.right)
            return f"({left_str} {self.op} {right_str})"
    
    def to_graph(
        self,
        op_mapping: Dict[str, int],
        feature_mapping: Dict[str, int],
        constant_mapping: Optional[Dict[float, int]] = None,
        graph_format_version: int = 1,
    ) -> Tuple[np.ndarray, List[Tuple[int, int]], Optional[List[int]]]:
        """Convert to graph representation with position encoding.
        
        Node features: [type, id, position]
        - type: NODE_TYPE_OP (0), NODE_TYPE_VAR (1)
        - id: operator ID or feature ID
        - position: POSITION_ROOT (0), POSITION_LEFT (1), or POSITION_RIGHT (2)
        
        Position encoding enables the model to distinguish between left and right
        operands, which is critical for non-commutative operations (a-b ≠ b-a).
        
        Note: Numeric constants are not supported - expressions should only use features.
        """
        nodes = []
        edges = []
        edge_types = [] if graph_format_version >= 2 else None
        
        class ConversionError(Exception):
            """Raised when feature or operator not found in mapping"""
            pass
        
        def _build_graph(
            node: ParseTreeNode,
            parent_idx: Optional[int] = None,
            position: int = POSITION_ROOT,
            depth: int = 0
        ) -> int:
            node_idx = len(nodes)
            
            if node.is_leaf():
                # Variable node with position encoding (constants not supported)
                feature_id = feature_mapping.get(node.value)
                if feature_id is None:
                    logging.warning(f"Feature '{node.value}' not in feature_mapping, cannot convert expression")
                    raise ConversionError(f"Unknown feature: {node.value}")
                if graph_format_version >= 2:
                    nodes.append([NODE_TYPE_VAR, feature_id, position, depth])
                else:
                    nodes.append([NODE_TYPE_VAR, feature_id, position])
            else:
                # Operator node with position encoding
                op_id = op_mapping.get(node.op)
                if op_id is None:
                    logging.warning(f"Operator '{node.op}' not in op_mapping, cannot convert expression")
                    raise ConversionError(f"Unknown operator: {node.op}")
                if graph_format_version >= 2:
                    nodes.append([NODE_TYPE_OP, op_id, position, depth])
                else:
                    nodes.append([NODE_TYPE_OP, op_id, position])
                
                # Process children with position encoding
                if node.left:
                    left_idx = _build_graph(node.left, node_idx, POSITION_LEFT, depth + 1)
                    edges.append((node_idx, left_idx))
                    edges.append((left_idx, node_idx))
                    if edge_types is not None:
                        edge_types.extend([EDGE_PARENT_TO_LEFT, EDGE_LEFT_TO_PARENT])
                
                if node.right:
                    right_idx = _build_graph(node.right, node_idx, POSITION_RIGHT, depth + 1)
                    edges.append((node_idx, right_idx))
                    edges.append((right_idx, node_idx))
                    if edge_types is not None:
                        edge_types.extend([EDGE_PARENT_TO_RIGHT, EDGE_RIGHT_TO_PARENT])
            
            if parent_idx is not None:
                # Parent edges are added by parent
                pass
            
            return node_idx
        
        try:
            _build_graph(self)
        except ConversionError:
            # Return empty arrays to signal failure
            if graph_format_version >= 2:
                return np.array([], dtype=np.int64).reshape(0, 4), [], []
            return np.array([], dtype=np.int64).reshape(0, 3), [], None
        
        if graph_format_version >= 2:
            return np.array(nodes, dtype=np.int64), edges, edge_types or []
        return np.array(nodes, dtype=np.int64), edges, None


def string_to_data_obj(
    expr_str: str,
    op_mapping: Dict[str, int],
    feature_mapping: Dict[str, int],
    constant_mapping: Optional[Dict[float, int]] = None,
    graph_format_version: int = 1,
) -> Optional[Data]:
    """Convert expression string to PyTorch Geometric Data object.
    
    Args:
        expr_str: Expression string to convert
        op_mapping: Operator to ID mapping
        feature_mapping: Feature name to ID mapping
        constant_mapping: Deprecated, kept for API compatibility (ignored)
    
    Note: Numeric constants are not supported - expressions with constants will fail to parse.
    """
    parser = ExpressionParser()
    tree = parser.parse(expr_str)
    
    if tree is None:
        return None
    
    try:
        nodes, edges, edge_types = tree.to_graph(
            op_mapping, feature_mapping, constant_mapping, graph_format_version
        )
        
        # Check if conversion failed (empty nodes array)
        if nodes.size == 0:
            logging.debug(f"Failed to convert expression to graph: {expr_str}")
            return None
        
        # Convert to PyTorch tensors
        x = torch.tensor(nodes, dtype=torch.long)
        
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
        
        if graph_format_version >= 2 and edge_types is not None:
            edge_attr = torch.tensor(edge_types, dtype=torch.long)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return Data(x=x, edge_index=edge_index)
        
    except Exception as e:
        logging.debug(f"Failed to convert expression to graph: {e}")
        return None


def data_obj_to_string(data: Data, op_mapping_inv: Dict[int, str], feature_mapping_inv: Dict[int, str],
                       constant_mapping_inv: Optional[Dict[int, float]] = None) -> Optional[str]:
    """Convert PyTorch Geometric Data object back to expression string.
    
    Supports both old format (2 columns: [type, id]) and new format (3 columns: [type, id, position]).
    When position encoding is available, uses it to correctly order left/right children.
    
    Args:
        data: PyTorch Geometric Data object
        op_mapping_inv: ID to operator mapping
        feature_mapping_inv: ID to feature name mapping
        constant_mapping_inv: Deprecated, kept for API compatibility (ignored)
    
    Note: Only NODE_TYPE_OP and NODE_TYPE_VAR are supported (no constants).
    """
    if data.x.size(0) == 0:
        return None
    
    # Check if position encoding is available (3 columns)
    has_position = data.x.size(1) >= 3
    
    # Build adjacency list
    adj_list = {i: [] for i in range(data.x.size(0))}
    for i in range(data.edge_index.size(1)):
        src, dst = data.edge_index[0, i].item(), data.edge_index[1, i].item()
        adj_list[src].append(dst)
    
    def build_expr(node_idx: int, visited: set) -> Optional[str]:
        if node_idx in visited:
            return None
        visited.add(node_idx)
        
        node_type = data.x[node_idx, 0].item()
        specific_id = data.x[node_idx, 1].item()
        
        if node_type == NODE_TYPE_VAR:
            # Variable leaf node
            return feature_mapping_inv.get(specific_id, f"var_{specific_id}")
        
        elif node_type == NODE_TYPE_OP:
            # Operator node - find children
            children = [n for n in adj_list[node_idx] if n not in visited]
            
            if len(children) >= 2:
                if has_position:
                    # Use position encoding to correctly order children
                    left_children = [n for n in children if data.x[n, 2].item() == POSITION_LEFT]
                    right_children = [n for n in children if data.x[n, 2].item() == POSITION_RIGHT]
                    
                    if left_children and right_children:
                        left_expr = build_expr(left_children[0], visited)
                        right_expr = build_expr(right_children[0], visited)
                    else:
                        # Fallback to sorted order if position not found
                        children_sorted = sorted(children)
                        left_expr = build_expr(children_sorted[0], visited)
                        right_expr = build_expr(children_sorted[1], visited)
                else:
                    # Old format: use sorted order
                    children_sorted = sorted(children)
                    left_expr = build_expr(children_sorted[0], visited)
                    right_expr = build_expr(children_sorted[1], visited)
                
                if left_expr and right_expr:
                    op = op_mapping_inv.get(specific_id, "+")
                    return f"({left_expr} {op} {right_expr})"
        
        return None
    
    # Try starting from different nodes to find valid expression
    for start_node in range(data.x.size(0)):
        if data.x[start_node, 0].item() == NODE_TYPE_OP:  # Start from operator nodes
            expr = build_expr(start_node, set())
            if expr:
                return expr
    
    # If no valid expression found from operators, try from node 0
    return build_expr(0, set())


def validate_expression(expr_str: str, lab_features: List[str]) -> bool:
    """Validate that expression uses only available features"""
    # Extract all potential variable names from expression
    # Remove operators and parentheses
    cleaned = expr_str
    for op in OPERATORS + ["(", ")"]:
        cleaned = cleaned.replace(op, " ")
    
    # Get tokens
    tokens = cleaned.split()
    
    # Check each token
    for token in tokens:
        if token and not any(token == feat or token.startswith(feat + "_") for feat in lab_features):
            # Check if it's a numeric constant
            try:
                float(token)
            except ValueError:
                return False
    
    return True


def sanitize_feature_name(name: str) -> str:
    """Sanitize feature name to be a valid Python identifier
    
    Used across the codebase to ensure consistent feature naming.
    """
    name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    name = name.strip('_')
    if name and name[0].isdigit():
        name = '_' + name
    name = re.sub(r'_+', '_', name)
    if name in ['def', 'class', 'return', 'lambda']:
        return name + "_var"
    return name if name else "feature"


try:
    from numba import jit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    def jit(*_args, **_kwargs):  # type: ignore
        def _decorator(fn):
            return fn
        return _decorator


@jit(nopython=True, cache=True)
def _calculate_auc_numba_fast(values: np.ndarray, target: np.ndarray) -> float:
    """Numba-accelerated rank-sum AUC with max(auc, 1-auc)."""
    mask = ~(np.isnan(values) | np.isnan(target))
    if np.sum(mask) < 10:
        return 0.5

    values_clean = values[mask]
    target_clean = target[mask]
    n = len(values_clean)

    n1 = 0
    for i in range(n):
        if target_clean[i] == 1.0:
            n1 += 1
    n0 = n - n1
    if n1 == 0 or n0 == 0:
        return 0.5

    sort_idx = np.argsort(values_clean)
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j < n - 1 and values_clean[sort_idx[j]] == values_clean[sort_idx[j + 1]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[sort_idx[k]] = avg_rank
        i = j + 1

    r1 = 0.0
    for i in range(n):
        if target_clean[i] == 1.0:
            r1 += ranks[i]

    u = r1 - n1 * (n1 + 1) / 2.0
    auc = u / (n1 * n0)
    return auc if auc >= 0.5 else 1.0 - auc


def calculate_univariate_auc(values_np: np.ndarray, target_np: np.ndarray) -> float:
    """Calculate univariate AUC using max(auc, 1-auc) for direction-agnostic signal."""
    if values_np is None or target_np is None:
        return 0.5
    values_arr = np.asarray(values_np, dtype=np.float64)
    target_arr = np.asarray(target_np, dtype=np.float64)
    try:
        if NUMBA_AVAILABLE:
            return float(_calculate_auc_numba_fast(values_arr, target_arr))
        raise RuntimeError("Numba unavailable")
    except Exception:
        try:
            from sklearn.metrics import roc_auc_score
            mask_np = ~(np.isnan(values_arr) | np.isnan(target_arr))
            if mask_np.sum() < 10:
                return 0.5
            auc = roc_auc_score(target_arr[mask_np], values_arr[mask_np])
            return max(auc, 1.0 - auc)
        except Exception as e:
            logging.debug(f"Error in AUC calculation: {e}")
            return 0.5


def calculate_information_gain_numpy(values_np: np.ndarray, target_np: np.ndarray) -> float:
    """Legacy IG name; returns univariate AUC."""
    return calculate_univariate_auc(values_np, target_np)


def convert_llm_expressions_to_gat_format(
    expressions_csv_path: str,
    op_mapping: Dict[str, int],
    feature_mapping: Dict[str, int],
    target_ig_column: str = "train_univariate_auc",
    label_keys: Optional[List[str]] = None,
    constant_mapping: Optional[Dict[float, int]] = None,
    graph_format_version: int = 1,
) -> Tuple[List[Tuple[Data, torch.Tensor]], int, int]:
    """
    Convert LLM-generated expressions from CSV to GAT training format.
    
    This function reads expressions from a CSV file, parses them into graph
    structures, and creates (Data, label_tensor) tuples compatible with
    GAT training. This allows the GAT model to be trained on diverse
    expression sources including LLM-generated expressions.
    
    Note: Expressions containing numeric constants will fail to convert.
    Only expressions using features and operators are supported.
    
    Args:
        expressions_csv_path: Path to CSV file with columns 'expression' and AUC values
        op_mapping: Operator to ID mapping (e.g., {'+': 0, '-': 1, '*': 2, '/': 3})
        feature_mapping: Feature name to ID mapping (e.g., {'lab_RDW_mean': 0, ...})
        target_ig_column: Column name containing univariate AUC values (legacy name kept for API compatibility)
        label_keys: Optional list of label keys for multi-target; if None, uses [target_ig_column]
        constant_mapping: Deprecated, kept for API compatibility (ignored)
        graph_format_version: Graph format version (1 or 2)
    
    Returns:
        Tuple of (dataset_list, num_converted, num_failed)
        - dataset_list: List of (Data, label_tensor) tuples for GAT training
        - num_converted: Number of successfully converted expressions
        - num_failed: Number of expressions that failed conversion
    """
    import pandas as pd
    
    if not os.path.exists(expressions_csv_path):
        logging.warning(f"Expressions CSV not found: {expressions_csv_path}")
        return [], 0, 0
    
    try:
        df = pd.read_csv(expressions_csv_path)
    except Exception as e:
        logging.error(f"Failed to read expressions CSV: {e}")
        return [], 0, 0
    
    if 'expression' not in df.columns:
        logging.error(f"CSV missing 'expression' column: {expressions_csv_path}")
        return [], 0, 0
    
    if target_ig_column not in df.columns:
        # Prefer AUC-native columns; fall back to legacy IG naming
        auc_cols = [c for c in df.columns if 'univariate_auc' in c.lower() or c.lower().endswith('_auc')]
        if not auc_cols:
            auc_cols = [c for c in df.columns if 'auc' in c.lower() and 'baseline' not in c.lower()]
        ig_cols = [c for c in df.columns if 'information_gain' in c.lower() or 'ig' in c.lower()]
        fallback_cols = auc_cols + ig_cols
        if fallback_cols:
            target_ig_column = fallback_cols[0]
            logging.info(f"Using '{target_ig_column}' as AUC column")
        else:
            logging.error(f"CSV missing AUC column '{target_ig_column}': {expressions_csv_path}")
            return [], 0, 0
    
    if label_keys is None:
        label_keys = [target_ig_column]
    
    dataset_list = []
    num_converted = 0
    num_failed = 0
    
    for idx, row in df.iterrows():
        expr_str = str(row['expression']).strip()
        
        # Get AUC value
        try:
            auc_value = float(row[target_ig_column])
            if np.isnan(auc_value):
                auc_value = 0.5
        except (ValueError, TypeError):
            auc_value = 0.5
        
        # Convert expression to graph (no constant support)
        data_obj = string_to_data_obj(
            expr_str,
            op_mapping,
            feature_mapping,
            constant_mapping,
            graph_format_version,
        )
        
        if data_obj is None or data_obj.x is None or data_obj.x.shape[0] == 0:
            num_failed += 1
            logging.debug(f"Failed to convert expression: {expr_str}")
            continue
        
        # Create label tensor (single AUC value reshaped to match GAT training format)
        label_tensor = torch.tensor([[auc_value]], dtype=torch.float32)
        
        dataset_list.append((data_obj, label_tensor))
        num_converted += 1
    
    logging.info(f"Converted {num_converted}/{len(df)} expressions from {os.path.basename(expressions_csv_path)} "
                 f"({num_failed} failed)")
    
    return dataset_list, num_converted, num_failed


def augment_gat_dataset_with_expressions(
    base_dataset_path: str,
    expression_csv_paths: List[str],
    output_path: str,
    stratified_sampling: bool = True,
    high_ig_oversample_factor: int = 5,
    high_ig_percentile: float = 95.0,
    filter_base_to_top_percentile: Optional[float] = None,
    llm_oversample_factor: int = 1
) -> bool:
    """
    Augment a GAT training dataset with expressions from multiple sources.
    
    This function loads an existing GAT training dataset (random trees) and
    augments it with expressions from LLM and other sources. It can:
    1. Filter base dataset to keep only top X% by AUC (filter_base_to_top_percentile)
    2. Oversample LLM expressions specifically (llm_oversample_factor)
    3. Apply stratified sampling to oversample high-AUC expressions
    
    Args:
        base_dataset_path: Path to original GAT dataset (.pt file)
        expression_csv_paths: List of CSV paths containing expressions to add
        output_path: Path to save augmented dataset
        stratified_sampling: If True, oversample high-AUC expressions from final dataset
        high_ig_oversample_factor: Legacy parameter (treated as high-AUC oversample factor)
        high_ig_percentile: Legacy parameter (treated as high-AUC percentile)
        filter_base_to_top_percentile: If set, keep only top X% of base expressions by AUC
        llm_oversample_factor: How many times to replicate LLM expressions
    
    Returns:
        True if augmentation succeeded, False otherwise
    """
    import os
    
    # Load base dataset
    if not os.path.exists(base_dataset_path):
        logging.error(f"Base dataset not found: {base_dataset_path}")
        return False
    
    try:
        # Handle PyTorch deserialization
        if hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
            from torch_geometric.data.data import DataEdgeAttr
            torch.serialization.add_safe_globals([DataEdgeAttr])
        
        loaded_payload = torch.load(base_dataset_path, map_location='cpu', weights_only=False)
        base_dataset = loaded_payload["dataset"]
        num_operators = loaded_payload["num_operators"]
        num_features = loaded_payload["num_features"]
        op_mapping = loaded_payload["op_mapping"]
        feature_mapping = loaded_payload["feature_mapping"]
        label_keys = loaded_payload.get("label_keys", [])
        graph_format_version = int(loaded_payload.get("graph_format_version", 1))
        num_edge_types = loaded_payload.get("num_edge_types")
        max_depth_used = loaded_payload.get("max_depth_used")
        
        logging.info(f"Loaded base dataset with {len(base_dataset)} expressions")
        logging.info(f"Graph format version: {graph_format_version}")
        if graph_format_version >= 2 and not num_edge_types:
            num_edge_types = NUM_EDGE_TYPES
    except Exception as e:
        logging.error(f"Failed to load base dataset: {e}")
        return False
    
    # Get AUC values from base dataset
    base_aucs = []
    for data_obj, label_tensor in base_dataset:
        try:
            auc_val = label_tensor[0, 0].item() if label_tensor.ndim > 1 else label_tensor[0].item()
            base_aucs.append(auc_val)
        except Exception:
            base_aucs.append(0.5)
    base_aucs = np.array(base_aucs)
    
    # Filter base dataset to keep only top X% by AUC
    if filter_base_to_top_percentile is not None and filter_base_to_top_percentile > 0:
        # Calculate threshold for top X% (top 2% means keeping AUC >= 98th percentile)
        threshold_percentile = 100.0 - filter_base_to_top_percentile
        auc_threshold = np.percentile(base_aucs, threshold_percentile)
        
        # Filter to keep only high-AUC expressions
        filtered_indices = np.where(base_aucs >= auc_threshold)[0]
        filtered_base = [base_dataset[i] for i in filtered_indices]
        filtered_aucs = base_aucs[filtered_indices]
        
        logging.info(f"Filtered base dataset: keeping top {filter_base_to_top_percentile}% "
                    f"({len(filtered_base)}/{len(base_dataset)} expressions, AUC >= {auc_threshold:.6f})")
        logging.info(f"  Filtered base AUCs: mean={np.mean(filtered_aucs):.6f}, "
                    f"max={np.max(filtered_aucs):.6f}, min={np.min(filtered_aucs):.6f}")
        
        all_data = list(filtered_base)
        base_filtered_count = len(filtered_base)
    else:
        all_data = list(base_dataset)
        base_filtered_count = len(base_dataset)
    
    # Convert and add expressions from each CSV
    added_from_llm = 0
    llm_data_all = []
    all_data_flags = [False] * len(all_data)
    
    for csv_path in expression_csv_paths:
        if not os.path.exists(csv_path):
            logging.warning(f"Expression CSV not found, skipping: {csv_path}")
            continue
        
        llm_data, num_converted, num_failed = convert_llm_expressions_to_gat_format(
            csv_path,
            op_mapping,
            feature_mapping,
            graph_format_version=graph_format_version,
        )
        llm_data_all.extend(llm_data)
        added_from_llm += num_converted
        
        # Collect AUCs from LLM expressions for logging
        if llm_data:
            llm_aucs = [t[1][0, 0].item() for t in llm_data]
            logging.info(f"  LLM expression AUCs: mean={np.mean(llm_aucs):.6f}, "
                        f"max={np.max(llm_aucs):.6f}, min={np.min(llm_aucs):.6f}")
    
    # Apply LLM-specific oversampling
    if llm_oversample_factor > 1 and llm_data_all:
        original_llm_count = len(llm_data_all)
        llm_data_all = llm_data_all * llm_oversample_factor
        logging.info(f"Oversampled LLM expressions: {original_llm_count} -> {len(llm_data_all)} "
                    f"(factor={llm_oversample_factor})")
    
    all_data.extend(llm_data_all)
    all_data_flags.extend([True] * len(llm_data_all))
    logging.info(f"Added {len(llm_data_all)} LLM expressions to dataset (from {added_from_llm} unique)")
    total_before_strat = len(all_data)
    llm_before_strat = sum(all_data_flags)
    llm_pct_before_strat = (llm_before_strat / total_before_strat * 100.0) if total_before_strat else 0.0
    logging.info(
        "Dataset composition pre-stratified: base_filtered=%d, llm_total=%d, total=%d, llm_pct=%.2f%%",
        base_filtered_count,
        llm_before_strat,
        total_before_strat,
        llm_pct_before_strat,
    )
    
    # Apply stratified sampling to oversample high-AUC expressions (optional)
    high_auc_oversample_factor = high_ig_oversample_factor
    high_auc_percentile = high_ig_percentile
    if stratified_sampling and len(all_data) > 0 and high_auc_oversample_factor > 1:
        # Get all AUCs
        all_aucs = []
        for data_obj, label_tensor in all_data:
            try:
                auc_val = label_tensor[0, 0].item() if label_tensor.ndim > 1 else label_tensor[0].item()
                all_aucs.append(auc_val)
            except Exception:
                all_aucs.append(0.5)
        
        all_aucs = np.array(all_aucs)
        auc_threshold = np.percentile(all_aucs, high_auc_percentile)
        high_auc_mask = all_aucs >= auc_threshold
        num_high_auc = np.sum(high_auc_mask)
        
        logging.info(f"Stratified sampling: {num_high_auc} expressions above {high_auc_percentile}th percentile "
                    f"(AUC >= {auc_threshold:.6f})")
        
        # Collect high-AUC expressions
        high_auc_data = [all_data[i] for i in range(len(all_data)) if high_auc_mask[i]]
        high_auc_flags = [all_data_flags[i] for i in range(len(all_data)) if high_auc_mask[i]]
        
        # Oversample high-AUC expressions
        oversampled = high_auc_data * (high_auc_oversample_factor - 1)
        oversampled_flags = high_auc_flags * (high_auc_oversample_factor - 1)
        all_data.extend(oversampled)
        all_data_flags.extend(oversampled_flags)
        
        logging.info(f"Added {len(oversampled)} oversampled high-AUC expressions "
                    f"(factor={high_auc_oversample_factor})")
    
    # Calculate and log final composition
    llm_total_final = sum(all_data_flags)
    llm_percentage = (llm_total_final / len(all_data)) * 100 if all_data else 0
    logging.info(
        "Final dataset composition: base_filtered=%d, llm_total=%d, total=%d, llm_pct=%.2f%%",
        base_filtered_count,
        llm_total_final,
        len(all_data),
        llm_percentage,
    )

    if graph_format_version >= 2 and max_depth_used is None:
        try:
            max_depth = 0
            for data_obj, _ in all_data:
                if data_obj is None or data_obj.x is None or data_obj.x.shape[1] < 4:
                    continue
                max_depth = max(max_depth, int(data_obj.x[:, 3].max().item()))
            max_depth_used = max_depth
            logging.info(f"Computed max_depth_used from augmented data: {max_depth_used}")
        except Exception as e:
            logging.warning(f"Failed to compute max_depth_used: {e}")
    
    # Save augmented dataset
    try:
        augmented_payload = {
            "dataset": all_data,
            "num_operators": num_operators,
            "num_features": num_features,
            "op_mapping": op_mapping,
            "feature_mapping": feature_mapping,
            "label_keys": label_keys,
            "graph_format_version": graph_format_version,
            "num_edge_types": num_edge_types,
            "max_depth_used": max_depth_used,
            "augmentation_info": {
                "base_count": len(base_dataset),
                "base_filtered_count": base_filtered_count,
                "filter_base_to_top_percentile": filter_base_to_top_percentile,
                "llm_added": added_from_llm,
                "llm_oversample_factor": llm_oversample_factor,
                "llm_total_after_oversample": len(llm_data_all),
                "final_count": len(all_data),
                "llm_percentage": llm_percentage,
                "stratified_sampling": stratified_sampling,
                "high_auc_oversample_factor": high_auc_oversample_factor if stratified_sampling else None,
                "high_auc_percentile": high_auc_percentile if stratified_sampling else None,
                # Legacy keys
                "high_ig_oversample_factor": high_auc_oversample_factor if stratified_sampling else None,
                "high_ig_percentile": high_auc_percentile if stratified_sampling else None
            }
        }
        
        torch.save(augmented_payload, output_path)
        logging.info(f"Saved augmented dataset ({len(all_data)} expressions) to {output_path}")
        return True
        
    except Exception as e:
        logging.error(f"Failed to save augmented dataset: {e}")
        return False
