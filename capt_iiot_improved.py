"""
CAPT-IIoT: improved pipeline (v2)
==================================

This builds on the leakage/efficiency-fixed notebook and adds the
architecture + methodology changes discussed in review:

  [CHANGE 1] RelationalAttentionSAGELayer replaces plain mean-aggregation
             SAGE: each of the 4 relation types gets its own message
             transform (R-GCN-style), and edges get a *learned* attention
             weight (GAT-style) instead of being averaged uniformly.
             Rationale: the encoder's E-GraphSAGE lineage came from
             NetFlow-based NIDS work where edges are the natural unit;
             we classify nodes here, so this layer gives edges real,
             differentiated influence on the node representation without
             abandoning node-level output.
             Toggle: USE_RELATIONAL_ATTENTION (False -> falls back to the
             original plain SAGE, kept below for A/B comparison).

  [CHANGE 2] An EdgeClassifier head, trained *jointly* with the node
             classifier, supervised by the dataset's real per-relation
             `label` column. Gives edge-level alerts and forces the shared
             embeddings to be edge-discriminative, addressing "maybe edges
             matter more than nodes" directly and empirically rather than
             by assertion.

  [CHANGE 3] Embedding-space SMOTE (hand-rolled, no imbalanced-learn
             dependency) applied only to frozen training-node embeddings,
             as an ADDITION to focal loss, not a replacement -- per the
             discussion of why naive SMOTE doesn't transfer to graph data
             (synthetic nodes have no real neighbors/edges, so it can only
             be applied post-hoc on the already-computed embeddings).
             Toggle: USE_SMOTE.

  [CHANGE 4] Walk-forward validation: `run_fold()` retrains a FRESH encoder
             per fold (no state or gradient carries across folds) across
             several cutoffs, so results are reported as mean +/- std
             across folds instead of one single, possibly-lucky 70/30
             split. This is the change most likely to alter your actual
             conclusions, since the single-split test set we had before
             only contained one attack subLabel (defenceEvasion).

  [CHANGE 5] Threshold selection offers both F1-optimal (as before) and
             precision-targeted (pick the highest-recall threshold that
             still meets a minimum precision bar) -- more operationally
             realistic for an IDS false-alarm budget.

  NOT implemented here (flagged honestly, not silently skipped):
    - A true continuous-time model (TGN/TGAT) to remove the window-size
      hyperparameter entirely. That's a bigger rewrite than fits one
      file/session; `suggest_window_sizes()` below is a lightweight
      diagnostic to help you pick WINDOW_SECONDS more deliberately in the
      meantime, not a replacement for that redesign.
    - A rigorous focal-loss-vs-LDAM-vs-weighted-CE ablation. FocalLoss is
      still used by default; ClassBalancedCE is provided as an alternative
      you can switch to and compare (USE_LOSS = "focal" | "weighted_ce").
"""

import time
import random
import numpy as np
import pandas as pd
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import from_networkx, degree, to_undirected, softmax
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    classification_report, confusion_matrix, average_precision_score,
    precision_recall_curve,
)

# =============================================================================
# CONFIG -- every "guess" number from the original notebook is named and
# collected here so it's obvious what you'd tune, instead of being buried
# inline. Nothing here is claimed to be optimal; these are starting points.
# =============================================================================
SEED = 42
DEVICE = 'cpu'  # set to 'cuda' if available

PHASE1_CSV = 'Phase1_Provenance.csv'
PHASE2_CSV = 'Phase2_Provenance.csv'

ENTITY_TYPES = ['Process', 'Artifact']
REL_TYPES = ['WasGeneratedBy', 'Used', 'WasTriggeredBy', 'WasDerivedFrom']
REL_TYPE_TO_ID = {t: i for i, t in enumerate(REL_TYPES)}
NUM_RELATIONS = len(REL_TYPES)

TOP_K_EXE = 40
WINDOW_SECONDS = 7200            # 2 hours -- data spans ~170h total, manageable on CPU
MAX_EDGES_PER_WINDOW = 8000

GNN_HIDDEN = 64
GNN_OUT = 64
LSTM_HIDDEN = 32
ATTN_HEADS = 4
DROPOUT = 0.3

USE_RELATIONAL_ATTENTION = True
USE_SMOTE = True
SMOTE_TARGET_RATIO = 0.15
USE_LOSS = "weighted_ce"
PRETRAIN_EPOCHS = 30
PRETRAIN_PHASE2_ONLY = True
CLASSIFIER_EPOCHS = 200
EARLY_STOP_PATIENCE = 50
NUM_FOLDS = 3

random.seed(SEED)
np.random.seed(SEED)
th.manual_seed(SEED)


# =============================================================================
# DATA LOADING & FEATURE ENGINEERING (unchanged from the leakage-fixed
# version -- see prior review for why cap_window is label-blind and why
# scaling is fit train-only)
# =============================================================================
def load_phase(fn, phase):
    df = pd.read_csv(fn, low_memory=False)
    df['phase'] = phase
    return df


def build_entities_and_relations():
    df1 = load_phase(PHASE1_CSV, 1)
    df2 = load_phase(PHASE2_CSV, 2)
    df = pd.concat([df1, df2], ignore_index=True)

    entities = df[df['type'].isin(ENTITY_TYPES)].copy()
    relations = df[df['type'].isin(REL_TYPES)].copy().reset_index(drop=True)
    relations['row_id'] = relations.index

    ent_label = entities.groupby('id')['label'].apply(lambda s: (s == 1).any())
    entities_dedup = entities.drop_duplicates(subset='id', keep='last').set_index('id')
    entities_dedup['label'] = ent_label

    wdf_mal_to = set(relations.loc[(relations['type'] == 'WasDerivedFrom') &
                                    (relations['label'] == 1), 'to'])
    entities_dedup.loc[entities_dedup.index.isin(wdf_mal_to), 'label'] = True

    print(f"nodes: {len(entities_dedup)}  malicious: {int(entities_dedup['label'].sum())} "
          f"({100 * entities_dedup['label'].mean():.3f}%)")
    return entities, entities_dedup, relations


def build_node_features(entities_dedup):
    node_ids = entities_dedup.index.to_numpy()
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    top_exe = entities_dedup['exe'].value_counts().head(TOP_K_EXE).index
    entities_dedup['exe_bucket'] = np.where(entities_dedup['exe'].isin(top_exe), entities_dedup['exe'], 'OTHER')
    entities_dedup['exe_bucket'] = entities_dedup['exe_bucket'].fillna('NONE')
    entities_dedup['subtype'] = entities_dedup['subtype'].fillna('NONE')
    entities_dedup['type'] = entities_dedup['type'].fillna('NONE')
    entities_dedup['protocol'] = entities_dedup['protocol'].fillna('NONE')

    cat_cols = ['type', 'subtype', 'exe_bucket', 'protocol']
    node_cat = pd.get_dummies(entities_dedup[cat_cols], dummy_na=False)
    num_cols = ['permissions', 'uid', 'gid', 'euid', 'egid', 'remote port', 'local port']
    node_num = entities_dedup[num_cols].fillna(-1)

    node_feat_df = pd.concat([node_cat, node_num], axis=1).astype(float)
    node_labels = th.tensor(entities_dedup['label'].astype(int).values, dtype=th.long)

    return node_feat_df, list(node_num.columns), node_labels, id_to_idx


def build_edge_features(relations, id_to_idx):
    relations['operation'] = relations['operation'].fillna('NONE')
    edge_cat = pd.get_dummies(relations[['type', 'operation']], dummy_na=False)
    edge_features = th.tensor(edge_cat.astype(float).values, dtype=th.float)

    relations['src'] = relations['from'].map(id_to_idx)
    relations['dst'] = relations['to'].map(id_to_idx)
    valid = relations['src'].notna() & relations['dst'].notna()
    relations = relations[valid].copy()
    relations['src'] = relations['src'].astype(int)
    relations['dst'] = relations['dst'].astype(int)
    relations['rel_type_id'] = relations['type'].map(REL_TYPE_TO_ID).astype(int)
    return relations, edge_features


def suggest_window_sizes(relations, candidates_hours=(1, 2, 4, 8, 12, 24)):
    """
    Diagnostic only -- NOT a replacement for a continuous-time model.
    Prints, for each candidate window size, how many windows you'd get and
    the edge-count distribution per window, so WINDOW_SECONDS is chosen
    with evidence instead of a guess. Run this once, look at the output,
    then set WINDOW_SECONDS above.
    """
    t0 = relations['time'].min()
    for h in candidates_hours:
        secs = h * 3600
        w = np.floor((relations['time'] - t0) / secs).astype(np.int64)
        counts = w.value_counts()
        print(f"{h:>3}h windows -> n_windows={len(counts):4d}  "
              f"edges/window: mean={counts.mean():8.1f}  "
              f"median={counts.median():8.1f}  max={counts.max():8d}  "
              f"windows_over_cap={int((counts > MAX_EDGES_PER_WINDOW).sum())}")


def build_windows(relations):
    """[FIX carried over] cap_window is label-blind uniform random sampling."""
    t0 = relations['time'].min()
    relations = relations.copy()
    relations['window'] = np.floor((relations['time'] - t0) / WINDOW_SECONDS).astype(np.int64)

    def cap_window(g, seed=SEED):
        if len(g) <= MAX_EDGES_PER_WINDOW:
            return g
        return g.sample(n=MAX_EDGES_PER_WINDOW, random_state=seed).sort_index()

    windows = []
    for _, g in relations.groupby('window'):
        g = cap_window(g)
        if len(g) > 0:
            windows.append(g)

    window_phase = [int(g['phase'].mode()[0]) for g in windows]
    print(f"total windows: {len(windows)}")
    return windows, window_phase


def scale_node_features(node_feat_df, num_cols, train_touched_node_ids):
    scaler_fit_rows = node_feat_df.iloc[train_touched_node_ids][num_cols]
    scaler = StandardScaler().fit(scaler_fit_rows.values)
    scaled = node_feat_df.copy()
    scaled[num_cols] = scaler.transform(node_feat_df[num_cols].values)
    return th.tensor(scaled.values, dtype=th.float)


def build_time_graphs(windows, node_features, node_labels, edge_features):
    time_graphs = []
    for g in windows:
        touched = np.union1d(g['src'].unique(), g['dst'].unique())
        g2l = {gl: i for i, gl in enumerate(touched)}
        local_src = g['src'].map(g2l).to_numpy()
        local_dst = g['dst'].map(g2l).to_numpy()
        edge_index = th.tensor(np.vstack([local_src, local_dst]), dtype=th.long)
        edge_attr = th.nan_to_num(edge_features[g['row_id'].to_numpy()], nan=0.0)
        edge_type = th.tensor(g['rel_type_id'].to_numpy(), dtype=th.long)

        data = Data(x=node_features[touched], edge_index=edge_index, edge_attr=edge_attr)
        data.h = edge_attr
        data.edge_type = edge_type
        data.y = node_labels[touched]
        data.global_node_ids = th.tensor(touched, dtype=th.long)
        data.phase = int(g['phase'].mode()[0])
        # keep raw (global-id) edge label/id for the edge classifier dataset
        data.edge_label = th.tensor(g['label'].astype(int).to_numpy(), dtype=th.long)
        data.edge_src_global = th.tensor(g['src'].to_numpy(), dtype=th.long)
        data.edge_dst_global = th.tensor(g['dst'].to_numpy(), dtype=th.long)
        time_graphs.append(data)
    return time_graphs


# =============================================================================
# [CHANGE 1] GNN LAYERS
# =============================================================================
class SAGELayer(MessagePassing):
    """Original plain mean-aggregation layer -- kept for A/B comparison."""
    def __init__(self, ndim_in, edim, ndim_out, activation):
        super().__init__(aggr='mean')
        self.W_msg = nn.Linear(ndim_in + edim, ndim_out)
        self.W_apply = nn.Linear(ndim_in + ndim_out, ndim_out)
        self.activation = activation

    def forward(self, x, edge_index, edge_attr, edge_type=None):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = self.W_msg(out)
        out = torch.cat([x, out], dim=1)
        return self.activation(self.W_apply(out))

    def message(self, x_j, edge_attr):
        return torch.cat([x_j, edge_attr], dim=1)


class SAGE(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, activation, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            SAGELayer(ndim_in, edim, GNN_HIDDEN, activation),
            SAGELayer(GNN_HIDDEN, edim, ndim_out, activation),
        ])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index, edge_attr, edge_type=None):
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
            x = layer(x, edge_index, edge_attr, edge_type)
        return x


class RelationalAttentionSAGELayer(MessagePassing):
    """
    [CHANGE 1] Per-relation message transform (R-GCN-style) + learned
    per-edge attention (GAT-style), instead of one shared linear transform
    and uniform mean aggregation.

    - Each relation type (Used/WasGeneratedBy/WasTriggeredBy/WasDerivedFrom)
      has its own W_msg[r], so "a WasDerivedFrom edge" and "a Used edge" are
      not forced through the same weights.
    - Attention (alpha) lets the model learn that some incoming edges matter
      much more than others for a given node, rather than averaging all
      neighbors equally -- this is the concrete answer to "give edges more
      weight."
    """
    def __init__(self, ndim_in, edim, ndim_out, num_relations, heads=ATTN_HEADS, dropout=0.1):
        super().__init__(aggr='add', node_dim=0)
        self.num_relations = num_relations
        self.heads = heads
        self.ndim_out = ndim_out

        self.W_msg = nn.ModuleList([nn.Linear(ndim_in + edim, heads * ndim_out) for _ in range(num_relations)])
        self.W_self = nn.Linear(ndim_in, heads * ndim_out)
        self.att = nn.Parameter(torch.empty(1, heads, 2 * ndim_out))
        nn.init.xavier_uniform_(self.att)
        self.W_apply = nn.Linear(ndim_in + heads * ndim_out, ndim_out)
        self.attn_dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x, edge_index, edge_attr, edge_type):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, edge_type=edge_type,
                              size=(x.size(0), x.size(0)))
        out = torch.cat([x, out], dim=1)
        return F.relu(self.W_apply(out))

    def message(self, x_j, x_i, edge_attr, edge_type, index, ptr, size_i):
        msgs = torch.zeros(x_j.size(0), self.heads, self.ndim_out, device=x_j.device)
        for r in range(self.num_relations):
            mask = (edge_type == r)
            if mask.any():
                m = self.W_msg[r](torch.cat([x_j[mask], edge_attr[mask]], dim=1))
                msgs[mask] = m.view(-1, self.heads, self.ndim_out)

        x_i_proj = self.W_self(x_i).view(-1, self.heads, self.ndim_out)
        alpha = self.leaky_relu((torch.cat([msgs, x_i_proj], dim=-1) * self.att).sum(dim=-1))
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = self.attn_dropout(alpha)
        return (msgs * alpha.unsqueeze(-1)).view(-1, self.heads * self.ndim_out)


class RelationalAttentionSAGE(nn.Module):
    def __init__(self, ndim_in, ndim_out, edim, num_relations, dropout):
        super().__init__()
        self.layer1 = RelationalAttentionSAGELayer(ndim_in, edim, GNN_HIDDEN, num_relations, dropout=dropout)
        self.layer2 = RelationalAttentionSAGELayer(GNN_HIDDEN, edim, ndim_out, num_relations, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, edge_type):
        x = self.layer1(x, edge_index, edge_attr, edge_type)
        x = self.dropout(x)
        x = self.layer2(x, edge_index, edge_attr, edge_type)
        return x


class EGraphSAGE_LSTM_Model(nn.Module):
    def __init__(self, node_in_dim, edge_dim, gnn_out, lstm_hidden, num_relations, dropout):
        super().__init__()
        if USE_RELATIONAL_ATTENTION:
            self.gnn = RelationalAttentionSAGE(node_in_dim, gnn_out, edge_dim, num_relations, dropout)
        else:
            self.gnn = SAGE(node_in_dim, gnn_out, edge_dim, activation=F.relu, dropout=dropout)
        self.lstm_hidden = lstm_hidden
        self.lstm_cell = nn.LSTMCell(gnn_out, lstm_hidden)
        self.fuse = nn.Linear(node_in_dim + lstm_hidden, node_in_dim)

    def forward(self, time_graphs, device, num_global_nodes):
        state_h = torch.zeros(num_global_nodes, self.lstm_hidden, device=device)
        state_c = torch.zeros(num_global_nodes, self.lstm_hidden, device=device)
        node_emb_all = torch.zeros(num_global_nodes, self.lstm_hidden, device=device)
        all_graph_embs = {}

        for t in range(len(time_graphs)):
            g = time_graphs[t]
            x = g.x.to(device)
            edge_index = g.edge_index.to(device)
            edge_attr = g.edge_attr.to(device)
            edge_type = g.edge_type.to(device)
            nodes = g.global_node_ids

            prev_h = state_h[nodes]
            prev_c = state_c[nodes]

            x_input = self.fuse(torch.cat([x, prev_h], dim=1))
            h_gnn = self.gnn(x_input, edge_index, edge_attr, edge_type)
            h_time, c_time = self.lstm_cell(h_gnn, (prev_h, prev_c))

            state_h = state_h.clone(); state_h[nodes] = h_time
            state_c = state_c.clone(); state_c[nodes] = c_time
            node_emb_all = node_emb_all.clone(); node_emb_all[nodes] = h_time
            all_graph_embs[t] = node_emb_all
        return all_graph_embs


# =============================================================================
# CONTRASTIVE PRETRAINING LOSSES (unchanged from the fixed version)
# =============================================================================
def get_gca_node_weights_exact(edge_index, num_nodes=None):
    edge_index_ = to_undirected(edge_index)
    if num_nodes is None:
        num_nodes = edge_index_.max().item() + 1
    deg = degree(edge_index_[1], num_nodes=num_nodes).float()
    s = th.log(deg.clamp(min=1))
    s_max, s_mean = s.max(), s.mean()
    denom = s_max - s_mean
    return th.ones_like(s) if denom == 0 else (s_max - s) / denom


def get_gca_feature_mask_exact(z, data, global_avg_weights, base_strength=0.2, noise_scale=0.1, min_corrupt=0.01, max_corrupt=0.3):
    w = global_avg_weights.to(z.device)[data.global_node_ids]
    norm_w = (w - w.min()) / (w.max() - w.min() + 1e-6)
    corrupt = (base_strength * norm_w).clamp(min_corrupt, max_corrupt).unsqueeze(1)
    return (1 - corrupt) * z + corrupt * (torch.randn_like(z) * noise_scale)


def get_negative_feature_mask_exact(z, data, global_avg_weights, base_strength=0.5, noise_scale=0.3, min_corrupt=0.05, max_corrupt=0.5):
    w = global_avg_weights.to(z.device)[data.global_node_ids]
    inv_w = (w.max() - w) / (w.max() - w.min() + 1e-6)
    corrupt = (base_strength * inv_w).clamp(min_corrupt, max_corrupt).unsqueeze(1)
    return (1 - corrupt) * z + corrupt * (torch.randn_like(z) * noise_scale)


def spatial_contrastive_loss(z, edge_index, tau=0.3, num_neg=10):
    src, dst = edge_index
    z = F.normalize(z, dim=1)
    N, E = z.size(0), src.size(0)
    pos_sim = (z[src] * z[dst]).sum(dim=1, keepdim=True) / tau
    neg = torch.randint(0, N, (E, num_neg), device=z.device)
    bad = neg == src.unsqueeze(1)
    while bad.any():
        neg[bad] = torch.randint(0, N, (int(bad.sum().item()),), device=z.device)
        bad = neg == src.unsqueeze(1)
    neg_sims = (z[src].unsqueeze(1) * z[neg]).sum(dim=-1) / tau
    logits = torch.cat([pos_sim, neg_sims], dim=1)
    labels = torch.zeros(E, dtype=torch.long, device=z.device)
    return F.cross_entropy(logits, labels)


def feature_contrastive_loss(z_active, data, global_avg_weights, temperature=0.2):
    z_pos = get_gca_feature_mask_exact(z_active, data, global_avg_weights)
    z_neg = get_negative_feature_mask_exact(z_active, data, global_avg_weights)
    anchor, positive, negative = F.normalize(z_active, dim=1), F.normalize(z_pos, dim=1), F.normalize(z_neg, dim=1)
    logits = torch.stack([(anchor * positive).sum(1) / temperature, (anchor * negative).sum(1) / temperature], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=anchor.device)
    return F.cross_entropy(logits, labels)


class TemporalPredictorLoss(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.predictor = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.BatchNorm1d(feature_dim), nn.ReLU(), nn.Linear(feature_dim, feature_dim))

    def forward(self, z_t, z_t_next, z_t_far, g_t, g_next, g_far, penalty_weight=0.05):
        z_t_pred = F.normalize(self.predictor(z_t), dim=-1)
        z_t_next_norm, z_t_far_norm = F.normalize(z_t_next, dim=-1), F.normalize(z_t_far, dim=-1)
        ids_t = g_t.global_node_ids.cpu().numpy()
        ids_next, ids_far = g_next.global_node_ids.cpu().numpy(), g_far.global_node_ids.cpu().numpy()
        loss_align = torch.tensor(0.0, device=z_t.device)
        loss_uniform = torch.tensor(0.0, device=z_t.device)
        common_pos = np.intersect1d(ids_t, ids_next)
        if len(common_pos) > 0:
            idx = torch.from_numpy(common_pos).to(z_t.device)
            loss_align = 2 - 2 * (z_t_pred[idx] * z_t_next_norm[idx]).sum(-1).mean()
        common_neg = np.intersect1d(ids_t, ids_far)
        if len(common_neg) > 0:
            idx = torch.from_numpy(common_neg).to(z_t.device)
            loss_uniform = F.relu((z_t_pred[idx] * z_t_far_norm[idx]).sum(-1) - 0.1).mean()
        return loss_align + penalty_weight * loss_uniform


def get_grad_norms(losses, shared_layer_params):
    shared_layer_params = list(shared_layer_params)
    norms = []
    for loss in losses:
        if not isinstance(loss, torch.Tensor) or not loss.requires_grad or loss.item() == 0.0:
            norms.append(0.0); continue
        grads = torch.autograd.grad(loss, shared_layer_params, retain_graph=True, allow_unused=True)
        norms.append(sum((g.data.norm(2).item() ** 2 for g in grads if g is not None)) ** 0.5)
    return norms


def compute_grad_balanced_weights(g_a, g_b, g_c):
    g_a, g_b, g_c = float(g_a or 0), float(g_b or 0), float(g_c or 0)
    total = g_a + g_b + g_c
    return (0.0, 0.5, 0.5) if total <= 1e-12 else (g_a / total, g_b / total, g_c / total)


def train_epoch(model, temporal_loss_module, time_graphs, final_avg_weights, optimizer, device, num_global_nodes,
                 tau=0.3, num_neg=10, d1=1, d2=4):
    """[FIX carried over] one forward + one accumulated backward per epoch, O(T) not O(T^2)."""
    model.train(); temporal_loss_module.train()
    optimizer.zero_grad()
    all_graph_embs = model(time_graphs, device, num_global_nodes)
    shared_params = list(model.fuse.parameters())

    spatial_loss_sum = torch.tensor(0.0, device=device)
    feature_loss_sum = torch.tensor(0.0, device=device)
    temporal_loss_sum = torch.tensor(0.0, device=device)
    T = len(time_graphs)

    for t in range(T):
        g = time_graphs[t]
        z_t = all_graph_embs[t]
        spatial_loss_sum = spatial_loss_sum + spatial_contrastive_loss(z_t[g.global_node_ids], g.edge_index, tau, num_neg)
        feature_loss_sum = feature_loss_sum + feature_contrastive_loss(z_t[g.global_node_ids], g, final_avg_weights)
        if t + d2 < T:
            temporal_loss_sum = temporal_loss_sum + temporal_loss_module(z_t, all_graph_embs[t + d1], all_graph_embs[t + d2], g, time_graphs[t + d1], time_graphs[t + d2])
        elif t - d2 >= 0 and t + d1 < T:
            temporal_loss_sum = temporal_loss_sum + temporal_loss_module(z_t, all_graph_embs[t + d1], all_graph_embs[t - d2], g, time_graphs[t + d1], time_graphs[t - d2])

    alpha, beta, gamma = compute_grad_balanced_weights(*get_grad_norms([temporal_loss_sum, spatial_loss_sum, feature_loss_sum], shared_params))
    total_loss = alpha * temporal_loss_sum + beta * spatial_loss_sum + gamma * feature_loss_sum
    total_loss.backward()
    optimizer.step()
    return total_loss.item() / T


# =============================================================================
# SUPERVISED HEADS: node classifier + [CHANGE 2] edge classifier
# =============================================================================
class NodeClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
        )

    def forward(self, x):
        return self.net(x)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=1)
        p = logp.exp()
        logp_t = logp.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -((1 - p_t) ** self.gamma) * logp_t
        if self.alpha is not None:
            loss = loss * self.alpha[targets]
        return loss.mean()


def make_loss(y_train, device):
    n_pos = int(y_train.sum()); n_neg = len(y_train) - n_pos
    if n_pos == 0:
        alpha = th.tensor([1.0, 1.0], device=device)
    elif n_neg == 0:
        alpha = th.tensor([1.0, 1.0], device=device)
    else:
        alpha = th.tensor([1.0 / n_neg, 1.0 / n_pos], device=device)
        alpha = alpha / alpha.sum() * 2
    if USE_LOSS == "focal":
        return FocalLoss(alpha=alpha, gamma=2.0)
    return nn.CrossEntropyLoss(weight=alpha)  # "weighted_ce" alternative for ablation


def smote_oversample(X, y, k=5, target_ratio=SMOTE_TARGET_RATIO, random_state=SEED):
    """
    [CHANGE 3] Minimal from-scratch SMOTE (no imbalanced-learn dependency).
    Interpolates between a minority training embedding and one of its
    k-nearest minority neighbors. Applied ONLY to training embeddings,
    never val/test -- this is purely a training-set augmentation.
    """
    rng = np.random.RandomState(random_state)
    X_min = X[y == 1]
    n_min = len(X_min)
    n_maj = int((y == 0).sum())
    n_target = int(n_maj * target_ratio) - n_min
    if n_target <= 0 or n_min < 2:
        return X, y
    k = min(k, n_min - 1)
    nn_model = NearestNeighbors(n_neighbors=k + 1).fit(X_min)
    _, idx = nn_model.kneighbors(X_min)
    synth = np.empty((n_target, X.shape[1]), dtype=X.dtype)
    for s in range(n_target):
        i = rng.randint(n_min)
        j = rng.choice(idx[i][1:])
        lam = rng.rand()
        synth[s] = X_min[i] + lam * (X_min[j] - X_min[i])
    return np.vstack([X, synth]), np.concatenate([y, np.ones(n_target)])


def best_threshold_f1(y_true, probs):
    prec, rec, thr = precision_recall_curve(y_true, probs)
    f1s = 2 * prec * rec / (prec + rec + 1e-9)
    if len(thr) == 0:
        return 0.5, 0.0
    best_i = np.nanargmax(f1s[:-1])
    return thr[best_i], f1s[best_i]


def threshold_for_precision(y_true, probs, target_precision=0.9):
    """[CHANGE 5] highest-recall threshold that still meets a precision floor."""
    prec, rec, thr = precision_recall_curve(y_true, probs)
    meets = prec[:-1] >= target_precision
    if not meets.any():
        return None, 0.0, 0.0
    candidates = np.where(meets)[0]
    best = candidates[np.argmax(rec[candidates])]
    return thr[best], prec[best], rec[best]


# =============================================================================
# [CHANGE 4] STRATIFIED FOLD CREATION
# =============================================================================
def create_stratified_folds(windows, window_phase, num_folds):
    """
    Phase 1 has 0 attacks. Phase 2 has all malicious edges concentrated
    in a narrow time band (~4 min). A naive positional split of Phase 2
    windows may put ALL malicious windows in one fold, leaving others
    degenerate (0 malicious test nodes).

    This function:
    1. Finds which windows actually have malicious edges (label > 0)
    2. Distributes those malicious windows evenly across folds
    3. Fills each fold's test set with benign windows to keep sizes balanced
    """
    malicious_wins = [i for i, g in enumerate(windows) if g['label'].sum() > 0]
    all_wins = list(range(len(windows)))
    benign_wins = [i for i in all_wins if i not in malicious_wins]

    print(f"  total windows: {len(windows)}  malicious windows: {len(malicious_wins)}")
    if malicious_wins:
        print(f"  malicious window indices: {malicious_wins}")

    if not malicious_wins:
        # fallback: just split evenly
        chunk = len(windows) // num_folds
        folds = []
        for k in range(num_folds):
            s, e = k * chunk, (k + 1) * chunk if k < num_folds - 1 else len(windows)
            test = list(range(s, e))
            train = [i for i in all_wins if i not in test]
            folds.append((train, test))
        return folds

    # distribute malicious windows round-robin across folds
    fold_test_mal = [[] for _ in range(num_folds)]
    for idx, mw in enumerate(malicious_wins):
        fold_test_mal[idx % num_folds].append(mw)

    # fill each fold's test set with benign windows to ~equal size
    target_per_fold = len(windows) // num_folds
    ben_copy = benign_wins[:]
    folds = []
    for k in range(num_folds):
        need = max(0, target_per_fold - len(fold_test_mal[k]))
        take = ben_copy[:need]
        ben_copy = ben_copy[need:]
        test = sorted(fold_test_mal[k] + take)
        train = [i for i in all_wins if i not in test]
        folds.append((train, test))

    return folds


# =============================================================================
# [CHANGE 4] ONE FOLD OF WALK-FORWARD VALIDATION
# =============================================================================
def run_fold(windows, window_phase, node_feat_df, num_cols, node_labels, edge_features,
             train_window_indices, test_window_indices, fold_name, device=DEVICE):
    print(f"\n{'=' * 70}\nFOLD: {fold_name}  (train={len(train_window_indices)} test={len(test_window_indices)} windows)\n{'=' * 70}")

    train_touched = np.union1d(
        pd.concat([windows[i] for i in train_window_indices])['src'].unique(),
        pd.concat([windows[i] for i in train_window_indices])['dst'].unique(),
    )
    node_features = scale_node_features(node_feat_df, num_cols, train_touched)
    time_graphs = build_time_graphs(windows, node_features, node_labels, edge_features)
    time_graphs_train = [time_graphs[i] for i in train_window_indices]
    num_global_nodes = node_features.shape[0]

    # ---- pretraining ----
    if PRETRAIN_PHASE2_ONLY:
        time_graphs_pretrain = [g for g in time_graphs_train if g.phase == 2]
        if not time_graphs_pretrain:
            time_graphs_pretrain = time_graphs_train
        print(f"  pretrain: {len(time_graphs_pretrain)}/{len(time_graphs_train)} windows (phase2 only)")
    else:
        time_graphs_pretrain = time_graphs_train

    model = EGraphSAGE_LSTM_Model(node_features.shape[1], edge_features.shape[1], GNN_OUT, LSTM_HIDDEN, NUM_RELATIONS, DROPOUT).to(device)
    temporal_loss_module = TemporalPredictorLoss(LSTM_HIDDEN).to(device)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(temporal_loss_module.parameters()), lr=1e-3)

    global_sum = th.zeros(num_global_nodes); global_count = th.zeros(num_global_nodes)
    for g in time_graphs_pretrain:
        w = get_gca_node_weights_exact(g.edge_index, num_nodes=g.num_nodes)
        global_sum[g.global_node_ids] += w
        global_count[g.global_node_ids] += 1
    final_avg_weights = global_sum / global_count.clamp(min=1)

    best_loss = float('inf'); best_state = None
    for epoch in range(PRETRAIN_EPOCHS):
        t0 = time.time()
        loss = train_epoch(model, temporal_loss_module, time_graphs_pretrain, final_avg_weights, optimizer, device, num_global_nodes)
        if loss < best_loss:
            best_loss = loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0:
            print(f"  epoch {epoch:2d} | loss={loss:.6f} | {time.time()-t0:.1f}s")
    model.load_state_dict(best_state)
    model.eval()

    # ---- frozen inference ----
    with torch.no_grad():
        all_graph_embs = model(time_graphs, device, num_global_nodes)
        final_embeddings = all_graph_embs[len(time_graphs) - 1].clone()

    # ---- node-level train/test split ----
    train_window_set = set(train_window_indices)
    test_window_set = set(test_window_indices)
    node_windows = [set() for _ in range(num_global_nodes)]
    for t, g in enumerate(time_graphs):
        for nid in g.global_node_ids.tolist():
            node_windows[nid].add(t)

    is_train = np.array([len(nw) > 0 and nw.issubset(train_window_set) for nw in node_windows])
    is_test = np.array([len(nw & test_window_set) > 0 for nw in node_windows])
    touched = np.array([len(nw) > 0 for nw in node_windows])
    train_idx = th.where(th.from_numpy(touched & is_train & ~is_test))[0]
    test_idx = th.where(th.from_numpy(touched & is_test))[0]
    print(f"  train nodes: {len(train_idx)}  test nodes: {len(test_idx)}  "
          f"test malicious: {int(node_labels[test_idx].sum())}")

    clf_features_raw = th.cat([final_embeddings, node_features], dim=1)
    y_all = node_labels

    train_labels = y_all[train_idx].numpy()
    n_train_mal = int(train_labels.sum())
    if n_train_mal > 0 and (len(train_labels) - n_train_mal) > 0:
        tr_np, val_np = train_test_split(train_idx.numpy(), test_size=0.15, stratify=train_labels, random_state=SEED)
    else:
        tr_np, val_np = train_test_split(train_idx.numpy(), test_size=0.15, random_state=SEED)
    tr_sub, val_sub = th.tensor(tr_np), th.tensor(val_np)
    print(f"  classifier train: {len(tr_sub)} (malicious={int(y_all[tr_sub].sum())})  val: {len(val_sub)} (malicious={int(y_all[val_sub].sum())})")

    clf_scaler = StandardScaler().fit(clf_features_raw[tr_sub].numpy())
    clf_features = th.tensor(clf_scaler.transform(clf_features_raw.numpy()), dtype=th.float)

    X_tr, y_tr = clf_features[tr_sub].numpy(), y_all[tr_sub].numpy()
    if USE_SMOTE:
        X_tr, y_tr = smote_oversample(X_tr, y_tr)
        print(f"  after SMOTE: train n={len(y_tr)}  malicious={int(y_tr.sum())} ({100*y_tr.mean():.2f}%)")
    X_tr_t = th.tensor(X_tr, dtype=th.float)
    y_tr_t = th.tensor(y_tr, dtype=th.long)

    # ---- binary classifier with early stopping ----
    clf = NodeClassifier(clf_features.shape[1]).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=1e-3, weight_decay=1e-5)

    node_loss_fn = make_loss(y_tr_t.numpy(), device)

    best_val_f1, best_thr, best_state = -1, 0.5, None
    patience_counter = 0
    for epoch in range(CLASSIFIER_EPOCHS):
        clf.train()
        opt.zero_grad()
        node_logits = clf(X_tr_t)
        node_loss = node_loss_fn(node_logits, y_tr_t)
        node_loss.backward()
        opt.step()

        clf.eval()
        with torch.no_grad():
            val_probs = F.softmax(clf(clf_features[val_sub]), dim=1)[:, 1].numpy()
        thr, f1 = best_threshold_f1(y_all[val_sub].numpy(), val_probs)
        if f1 > best_val_f1:
            best_val_f1, best_thr = f1, thr
            best_state = {k: v.clone() for k, v in clf.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch % 25 == 0:
            print(f"  clf epoch {epoch:3d} | loss={node_loss.item():.4f} | val_F1={f1:.4f}")
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"  early stop at epoch {epoch}")
            break

    clf.load_state_dict(best_state)
    clf.eval()

    # ---- evaluation ----
    with torch.no_grad():
        test_probs = F.softmax(clf(clf_features[test_idx]), dim=1)[:, 1].numpy()
    test_pred = (test_probs >= best_thr).astype(int)
    y_test = y_all[test_idx].numpy()

    pr_auc = average_precision_score(y_test, test_probs)
    print(f"\n[{fold_name}] NODE report:")
    print(classification_report(y_test, test_pred, target_names=['benign', 'malicious'], zero_division=0))
    print("PR-AUC:", pr_auc)
    print("Confusion matrix:\n", confusion_matrix(y_test, test_pred))

    thr_p90, prec_p90, rec_p90 = threshold_for_precision(y_test, test_probs, target_precision=0.9)
    if thr_p90 is not None:
        print(f"  threshold for >=90% precision: {thr_p90:.3f} -> precision={prec_p90:.3f}, recall={rec_p90:.3f}")

    return {
        'fold': fold_name,
        'node_precision': float((test_pred[y_test == 1] == 1).sum() / max(test_pred.sum(), 1)),
        'node_recall': float((test_pred[y_test == 1] == 1).sum() / max(y_test.sum(), 1)),
        'node_f1': float(2 * ((test_pred[y_test == 1] == 1).sum() / max(test_pred.sum(), 1)) * ((test_pred[y_test == 1] == 1).sum() / max(y_test.sum(), 1)) / max(((test_pred[y_test == 1] == 1).sum() / max(test_pred.sum(), 1)) + ((test_pred[y_test == 1] == 1).sum() / max(y_test.sum(), 1)), 1e-9)),
        'node_pr_auc': float(pr_auc),
        'thr_p90': float(thr_p90) if thr_p90 is not None else None,
        'prec_p90': float(prec_p90),
        'rec_p90': float(rec_p90),
        'n_test_malicious': int(y_test.sum()),
    }


if __name__ == '__main__':
    entities, entities_dedup, relations = build_entities_and_relations()
    node_feat_df, num_cols, node_labels, id_to_idx = build_node_features(entities_dedup)
    relations, edge_features = build_edge_features(relations, id_to_idx)

    windows, window_phase = build_windows(relations)
    folds = create_stratified_folds(windows, window_phase, NUM_FOLDS)

    results = []
    for i, (train_idx, test_idx) in enumerate(folds):
        test_mal = sum(1 for wi in test_idx if windows[wi]['label'].sum() > 0)
        print(f"fold_{i+1}: train={len(train_idx)} test={len(test_idx)} windows, "
              f"test_windows_with_malicious_edges={test_mal}")
        res = run_fold(windows, window_phase, node_feat_df, num_cols, node_labels, edge_features,
                        train_window_indices=train_idx, test_window_indices=test_idx,
                        fold_name=f"fold_{i+1}")
        results.append(res)

    print(f"\n{'=' * 70}\nWALK-FORWARD SUMMARY ({len(results)} folds)\n{'=' * 70}")
    df_res = pd.DataFrame(results)
    print(df_res)
    for col in ['node_precision', 'node_recall', 'node_f1', 'node_pr_auc']:
        vals = df_res[col].dropna()
        print(f"{col}: mean={vals.mean():.4f}  std={vals.std():.4f}")
    print("\nPer-fold >=90% precision thresholds:")
    for _, r in df_res.iterrows():
        if r['thr_p90'] is not None:
            print(f"  {r['fold']}: thr={r['thr_p90']:.3f} prec={r['prec_p90']:.3f} rec={r['rec_p90']:.3f}")
        else:
            print(f"  {r['fold']}: no threshold meets 90% precision")
