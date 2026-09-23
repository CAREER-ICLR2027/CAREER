"""SGAP hierarchy construction inherited from CVSearch; no search policy."""

import networkx as nx
import numpy as np
import torch
from skimage import graph
from skimage.measure import regionprops
from skimage.segmentation import slic
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


class ConstrainedTreeBuilder:
    def __init__(self, feature_map, n_atoms=400, pos_weight=2.0, split_threshold=0.3, keep_threshold=0.05, use_local_normalization=True,
                 use_silhouette_score=True):
        if isinstance(feature_map, torch.Tensor):
            self.feat = feature_map.detach().cpu().numpy()
        else:
            self.feat = feature_map

        self.C, self.H, self.W = self.feat.shape
        self.n_atoms = n_atoms
        self.pos_weight = pos_weight
        self.split_threshold = split_threshold
        self.keep_threshold = keep_threshold
        self.use_local_normalization = use_local_normalization
        self.use_silhouette_score = use_silhouette_score
        self.atom_labels, self.atom_features, self.atom_bboxes, self.adj_matrix = self._generate_atoms_and_graph()

    def _generate_atoms_and_graph(self):
        feat_tr = self.feat.transpose(1, 2, 0)
        feat_min, feat_max = feat_tr.min(), feat_tr.max()
        feat_norm = (feat_tr - feat_min) / (feat_max - feat_min + 1e-6)
        #SLIC
        atom_map = slic(feat_norm, n_segments=self.n_atoms, compactness=20, start_label=0, channel_axis=2)
        unique_labels = np.unique(atom_map)
        n_actual = len(unique_labels)
        semantic_features = np.zeros((n_actual, self.C), dtype=np.float32)
        spatial_features = np.zeros((n_actual, 2), dtype=np.float32)
        props = regionprops(atom_map + 1)
        atom_bboxes = []

        for i, prop in enumerate(props):
            y_slice, x_slice = prop.slice
            mask_local = prop.image
            feat_crop = self.feat[:, y_slice, x_slice]
            semantic_features[i] = feat_crop[:, mask_local].mean(axis=1)

            cy, cx = prop.centroid
            y_encoded = (cy / self.H) * self.pos_weight
            x_encoded = (cx / self.W) * self.pos_weight
            spatial_features[i] = [y_encoded, x_encoded]

            atom_bboxes.append(prop.bbox)

        final_features = np.concatenate([semantic_features, spatial_features], axis=1)

        # Connectivity is spatial adjacency, including zero-distance neighbors.
        rag = graph.RAG(atom_map, connectivity=2)
        rag.add_nodes_from(range(n_actual))
        adj_matrix = nx.adjacency_matrix(rag, nodelist=range(n_actual), weight=None)
        return atom_map, final_features, np.array(atom_bboxes), adj_matrix

    def _calc_overlap_cost(self, child_nodes):
        if len(child_nodes) < 2: return 0.0

        boxes = [n['bbox'] for n in child_nodes]
        total_area = sum([(b[2] - b[0]) * (b[3] - b[1]) for b in boxes])
        total_overlap = 0.0

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                ba, bb = boxes[i], boxes[j]
                iy1, ix1 = max(ba[0], bb[0]), max(ba[1], bb[1])
                iy2, ix2 = min(ba[2], bb[2]), min(ba[3], bb[3])
                inter = max(0, iy2 - iy1) * max(0, ix2 - ix1)
                total_overlap += inter

        if total_area == 0: return 0
        return total_overlap / total_area

    def _calc_region_complexity(self, atom_indices):
        if len(atom_indices) <= 1:
            return 0.0

        # semantic features (N_subset, C)
        features = self.atom_features[atom_indices, :self.C]
        norm = np.linalg.norm(features, axis=1, keepdims=True) + 1e-6
        feats_norm = features / norm
        mean_feat = feats_norm.mean(axis=0)
        mean_feat_norm = mean_feat / (np.linalg.norm(mean_feat) + 1e-6)

        # cosine similarity
        # shapes: (N, C) @ (C,) -> (N,)
        cosine_sims = feats_norm @ mean_feat_norm
        avg_sim = cosine_sims.mean()
        # Convert to complexity score
        score = max(0.0, 1.0 - avg_sim)

        avg_magnitude = norm.mean()
        score *= np.log1p(avg_magnitude)

        return score

    def build_tree(self, max_depth=3, min_splits=4, max_splits=8):
        all_indices = np.arange(len(self.atom_features))
        # Calculate the complexity of the global image
        global_complexity = self._calc_region_complexity(all_indices)

        root_node = {
            "depth": 0,
            "atom_indices": all_indices,
            "bbox": (0, 0, self.H, self.W),
            "children": [],
            "split_k": 1,
            "complexity": global_complexity,
            "node_id": "0",
            "parent": None
        }
        self._recursive_build(root_node, max_depth, min_splits, max_splits)
        return root_node

    def _recursive_build(self, parent_node, max_depth, min_splits, max_splits):
        if parent_node["depth"] >= max_depth:
            return

        indices = parent_node["atom_indices"]
        effective_min_k = min_splits
        if len(indices) < max(min_splits, 2):
            return

        # Semantic Pruning
        if parent_node["depth"] >= 1:
            depth_decay = 0.8 ** (parent_node["depth"] - 1)
            current_threshold = self.split_threshold * depth_decay

            complexity = parent_node.get("complexity", 0)
            if complexity < current_threshold:
                return

        raw_sub_features = self.atom_features[indices]
        semantic_raw = raw_sub_features[:, :self.C]
        if self.use_local_normalization:
            scaler = StandardScaler()
            enhanced_semantic = scaler.fit_transform(semantic_raw)
        else:
            enhanced_semantic = semantic_raw

        spatial_feats = raw_sub_features[:, self.C:]
        spatial_scale = 1.0 / (parent_node["depth"] + 1)
        sub_features_for_clustering = np.concatenate([enhanced_semantic, spatial_feats * spatial_scale], axis=1)

        sub_connectivity = self.adj_matrix[indices, :][:, indices]

        best_score = -float('inf')
        best_children = []
        best_k = effective_min_k
        best_labels = None
        limit_k = min(max_splits, len(indices))
        if limit_k < effective_min_k: return

        for k in range(effective_min_k, limit_k + 1):
            try:
                model = AgglomerativeClustering(
                    n_clusters=k,
                    connectivity=sub_connectivity,
                    linkage='ward'
                )
                labels = model.fit_predict(sub_features_for_clustering)

                if self.use_silhouette_score:
                    if k > 1 and len(indices) > k:
                        sil_score = silhouette_score(enhanced_semantic, labels)
                    else:
                        sil_score = -1.0
                else:
                    sil_score = 0.0

                # Calculate BBox overlap cost
                overlap_cost = self._calc_overlap_cost_for_labels(indices, labels, k)
                # --- Comprehensive scoring formula ---
                combined_score = sil_score - (1.5 * overlap_cost)
                if combined_score > best_score:
                    best_score = combined_score
                    best_k = k
                    best_labels = labels

            except ValueError:
                continue

        if best_score == -float('inf'): return
        best_children = []
        for lbl in range(best_k):
            child_indices_local = np.where(best_labels == lbl)[0]
            if len(child_indices_local) == 0: continue

            child_atom_indices = indices[child_indices_local]
            c_boxes = self.atom_bboxes[child_atom_indices]
            y1, x1 = np.min(c_boxes[:, 0]), np.min(c_boxes[:, 1])
            y2, x2 = np.max(c_boxes[:, 2]), np.max(c_boxes[:, 3])

            best_children.append({
                "atom_indices": child_atom_indices,
                "bbox": (y1, x1, y2, x2)
            })

        if not best_children: return

        # Calculate complexity and prune
        valid_children_data = []  # (child_data, complexity)

        for child_data in best_children:
            child_complexity = self._calc_region_complexity(child_data["atom_indices"])
            if child_complexity < self.keep_threshold:
                continue

            valid_children_data.append((child_data, child_complexity))

        # All child nodes have been pruned
        if not valid_children_data:
            return

        for i, (child_data, child_complexity) in enumerate(valid_children_data):
            current_node_id = f"{parent_node['node_id']}-{i}"

            child_node = {
                "depth": parent_node["depth"] + 1,
                "atom_indices": child_data["atom_indices"],
                "bbox": child_data["bbox"],
                "children": [],
                "split_k": best_k,
                "complexity": child_complexity,
                "node_id": current_node_id,
                "parent": parent_node
            }
            parent_node["children"].append(child_node)

            self._recursive_build(child_node, max_depth, min_splits, max_splits)

    def _calc_overlap_cost_for_labels(self, indices, labels, k):
        temp_children = []
        for lbl in range(k):
            child_indices_local = np.where(labels == lbl)[0]
            if len(child_indices_local) == 0: continue
            child_atom_indices = indices[child_indices_local]
            c_boxes = self.atom_bboxes[child_atom_indices]
            y1, x1 = np.min(c_boxes[:, 0]), np.min(c_boxes[:, 1])
            y2, x2 = np.max(c_boxes[:, 2]), np.max(c_boxes[:, 3])
            temp_children.append({"bbox": (y1, x1, y2, x2)})
        return self._calc_overlap_cost(temp_children)
