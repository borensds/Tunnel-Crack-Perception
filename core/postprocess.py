import numpy as np
import networkx as nx
from scipy import ndimage


class FractureAnalyzer:
    def __init__(
            self,
            pixel_scale=1.0,
            junction_remove_radius=1,
            connect_gap=12,
            collinear_theta_deg=25,
            alpha_main=20.0,
            alpha_branch=30.0,
            min_seg_pts=6,
    ):
        self.scale = pixel_scale
        self.junction_remove_radius = junction_remove_radius
        self.connect_gap = connect_gap
        self.collinear_theta_deg = collinear_theta_deg
        self.alpha_main = alpha_main
        self.alpha_branch = alpha_branch
        self.min_seg_pts = min_seg_pts

    def analyze_with_labels(self, mask, label_map, num_labels):
        dist_map = ndimage.distance_transform_edt(mask // 255)
        skeleton = (self._skeletonize(mask) > 0).astype(np.uint8)

        rows = []
        for label_id in range(1, num_labels):
            skel = ((skeleton > 0) & (label_map == label_id)).astype(np.uint8)
            if skel.sum() < self.min_seg_pts:
                continue

            # 1. 拓扑解耦：打断交叉点，获取简单线段
            decoupled = self._topology_decouple(skel, radius=self.junction_remove_radius)

            # 2. 特征提取与建图
            segments = self._extract_segments(decoupled, dist_map)
            segments = [s for s in segments if s["npts"] >= self.min_seg_pts]
            if not segments:
                continue

            # 3. 基于共线约束的线段邻接图构建与主干搜索
            seg_graph = self._build_segment_adjacency_graph(segments)
            main_seg_ids = self._longest_weighted_chain(seg_graph)

            for seg in segments:
                seg["role"] = "Branch"
            for sid in main_seg_ids:
                segments[sid]["role"] = "Main"

            # 4. PF-GAP 分级剪枝：根据长宽比剔除假性裂隙
            kept = []
            for seg in segments:
                alpha = self.alpha_main if seg["role"] == "Main" else self.alpha_branch
                if self._pf_gap_keep(seg["length"], seg["w_avg"], alpha=alpha):
                    kept.append(seg)

            main_pts = np.vstack([s["pts"] for s in kept if s["role"] == "Main"]) if any(
                s["role"] == "Main" for s in kept) else None
            if main_pts is None or len(main_pts) < 2:
                continue

            # 5. 亚像素级工程参数量化
            length = self._path_length(main_pts)
            w_vals = dist_map[main_pts[:, 0], main_pts[:, 1]] * 2.0
            w_avg = float(np.mean(w_vals)) if len(w_vals) else 0.0
            w_max = float(np.max(w_vals)) if len(w_vals) else 0.0
            angle = self._pca_orientation_deg(main_pts)
            cy, cx = np.mean(main_pts, axis=0)

            rows.append({
                "ID": label_id,
                "长度(mm)": round(length * self.scale, 2),
                "平均宽度(mm)": round(w_avg * self.scale, 2),
                "最大宽度(mm)": round(w_max * self.scale, 2),
                "倾角(°)": round(angle, 1),
                "中心X": int(cx),
                "中心Y": int(cy),
            })

        import pandas as pd
        return pd.DataFrame(rows)

    def _skeletonize(self, mask):
        from skimage.morphology import skeletonize
        return skeletonize(mask // 255).astype(np.uint8)

    def _topology_decouple(self, skel01, radius=1):
        deg = self._degree_map_8n(skel01)
        junction = (skel01 > 0) & (deg >= 3)

        if radius <= 0:
            out = skel01.copy()
            out[junction] = 0
            return out

        import cv2
        j = junction.astype(np.uint8) * 255
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        j_d = cv2.dilate(j, k) > 0

        out = skel01.copy()
        out[j_d] = 0
        return out

    def _degree_map_8n(self, skel01):
        import cv2
        kernel = np.ones((3, 3), np.uint8)
        nb = cv2.filter2D(skel01.astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)
        return nb - skel01

    def _extract_segments(self, decoupled01, dist_map):
        import cv2
        num, cc = cv2.connectedComponents(decoupled01.astype(np.uint8), connectivity=8)
        segs = []
        for cid in range(1, num):
            pts = np.argwhere(cc == cid)
            if len(pts) < 2:
                continue

            G = self._build_pixel_graph(pts)
            endpoints = [n for n, d in G.degree() if d == 1]
            if len(endpoints) >= 2:
                s, t = endpoints[0], endpoints[-1]
            else:
                s, t = self._tree_diameter_endpoints(G)

            try:
                path = nx.shortest_path(G, s, t, weight="weight")
            except Exception:
                path = list(G.nodes())

            path_pts = np.array(path, dtype=np.int32)
            length = self._path_length(path_pts)

            w_vals = dist_map[path_pts[:, 0], path_pts[:, 1]] * 2.0
            w_avg = float(np.mean(w_vals)) if len(w_vals) else 0.0

            dir_vec = self._pca_direction(path_pts)
            segs.append({
                "pts": path_pts,
                "endpoints": (np.array(s), np.array(t)),
                "length": float(length),
                "w_avg": float(w_avg),
                "dir": dir_vec,
                "npts": int(len(path_pts)),
            })
        return segs

    def _build_pixel_graph(self, pts):
        G = nx.Graph()
        pts_t = [tuple(p) for p in pts]
        S = set(pts_t)
        for (y, x) in pts_t:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx_ = y + dy, x + dx
                    if (ny, nx_) in S:
                        w = float(np.hypot(dy, dx))
                        G.add_edge((y, x), (ny, nx_), weight=w)
        return G

    def _tree_diameter_endpoints(self, G):
        nodes = list(G.nodes())
        if not nodes: return (0, 0), (0, 0)
        a = nodes[0]
        dist = nx.single_source_dijkstra_path_length(G, a, weight="weight")
        b = max(dist.items(), key=lambda kv: kv[1])[0]
        dist2 = nx.single_source_dijkstra_path_length(G, b, weight="weight")
        c = max(dist2.items(), key=lambda kv: kv[1])[0]
        return b, c

    def _path_length(self, pts_yx):
        if len(pts_yx) < 2: return 0.0
        d = np.diff(pts_yx.astype(np.float32), axis=0)
        return float(np.sum(np.hypot(d[:, 0], d[:, 1])))

    def _pca_direction(self, pts_yx):
        x = pts_yx[:, 1].astype(np.float32)
        y = pts_yx[:, 0].astype(np.float32)
        X = np.stack([x - x.mean(), y - y.mean()], axis=0)
        C = X @ X.T
        vals, vecs = np.linalg.eig(C)
        v = vecs[:, np.argmax(vals)]
        return v / (np.linalg.norm(v) + 1e-8)

    def _pca_orientation_deg(self, pts_yx):
        v = self._pca_direction(pts_yx)
        ang = abs(np.degrees(np.arctan2(float(v[1]), float(v[0]))))
        return float(180 - ang if ang > 90 else ang)

    def _build_segment_adjacency_graph(self, segments):
        G = nx.Graph()
        for i in range(len(segments)): G.add_node(i)
        th = np.deg2rad(self.collinear_theta_deg)

        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                si, sj = segments[i], segments[j]
                gap, _ = self._min_endpoint_gap(si["endpoints"], sj["endpoints"])
                if gap > self.connect_gap: continue

                cos = float(abs(np.dot(si["dir"], sj["dir"])))
                ang = float(np.arccos(np.clip(cos, -1.0, 1.0)))
                if ang > th: continue

                w = (si["length"] + sj["length"]) - 2.0 * gap - 10.0 * ang
                G.add_edge(i, j, weight=float(w))
        return G

    def _min_endpoint_gap(self, ep_i, ep_j):
        a1, a2 = ep_i
        b1, b2 = ep_j
        pairs = [(a1, b1), (a1, b2), (a2, b1), (a2, b2)]
        ds = [float(np.linalg.norm(p[0] - p[1])) for p in pairs]
        k = int(np.argmin(ds))
        return ds[k], pairs[k]

    def _longest_weighted_chain(self, seg_graph):
        if seg_graph.number_of_nodes() == 1:
            return [list(seg_graph.nodes())[0]]

        endpoints = [n for n, d in seg_graph.degree() if d <= 1]
        if not endpoints: endpoints = list(seg_graph.nodes())

        best_path, best_w = [], -1e18

        def dfs(cur, visited, w_acc, path):
            nonlocal best_w, best_path
            if w_acc > best_w:
                best_w = w_acc
                best_path = path[:]
            for nxt in seg_graph.neighbors(cur):
                if nxt in visited: continue
                w = seg_graph.edges[cur, nxt].get("weight", 0.0)
                visited.add(nxt)
                path.append(nxt)
                dfs(nxt, visited, w_acc + w, path)
                path.pop()
                visited.remove(nxt)

        for s in endpoints:
            dfs(s, {s}, 0.0, [s])
        return best_path

    def _pf_gap_keep(self, L_seg, W_avg, alpha):
        return (L_seg / (W_avg + 1e-6)) >= alpha