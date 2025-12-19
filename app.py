#!/usr/bin/env python3
"""
FACS Autogating - Auto-Gating avec modification par boutons
- Détection automatique robuste des canaux
- Auto-gating GMM
- Modification des gates par boutons
- Apprentissage des corrections
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from sklearn.mixture import GaussianMixture
from scipy.spatial import ConvexHull
from pathlib import Path as FilePath
import tempfile
import io
import json
import os
from datetime import datetime
import flowio
import re

st.set_page_config(page_title="FACS - Auto-Gating", page_icon="🔬", layout="wide")

LEARNED_PARAMS_FILE = "learned_gating_params.json"

st.markdown("""
    <style>
    .main-header { font-size: 1.8rem; color: #2c3e50; text-align: center; }
    .info-box { background: #e7f3ff; padding: 0.8rem; border-radius: 0.5rem; border-left: 4px solid #0066cc; margin: 0.5rem 0; }
    .gate-edit { background: #fff8e7; padding: 0.6rem; border-radius: 0.4rem; border: 1px solid #ffc107; margin: 0.3rem 0; }
    </style>
""", unsafe_allow_html=True)


def load_learned_params():
    if os.path.exists(LEARNED_PARAMS_FILE):
        try:
            with open(LEARNED_PARAMS_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {'n_corrections': 0, 'gates': {}}


def update_learned_params(gate_name, original_polygon, corrected_polygon):
    params = load_learned_params()
    if gate_name not in params['gates']:
        params['gates'][gate_name] = {'avg_adjustment': {'x': 0, 'y': 0}, 'n_samples': 0}
    gate_params = params['gates'][gate_name]
    orig_center = np.mean(original_polygon, axis=0)
    corr_center = np.mean(corrected_polygon, axis=0)
    dx, dy = float(corr_center[0] - orig_center[0]), float(corr_center[1] - orig_center[1])
    gate_params['n_samples'] += 1
    n = gate_params['n_samples']
    alpha = 2 / (n + 1)
    gate_params['avg_adjustment']['x'] = (1 - alpha) * gate_params['avg_adjustment']['x'] + alpha * dx
    gate_params['avg_adjustment']['y'] = (1 - alpha) * gate_params['avg_adjustment']['y'] + alpha * dy
    params['n_corrections'] += 1
    with open(LEARNED_PARAMS_FILE, 'w') as f:
        json.dump(params, f)
    return params


class FCSReader:
    def __init__(self, fcs_path):
        self.flow_data = flowio.FlowData(fcs_path)
        self.filename = FilePath(fcs_path).stem
        events = np.array(self.flow_data.events, dtype=np.float64)
        n_ch = self.flow_data.channel_count
        if events.ndim == 1:
            events = events.reshape(-1, n_ch)
        labels = []
        for i in range(1, n_ch + 1):
            pnn = self.flow_data.text.get(f'$P{i}N', '') or self.flow_data.text.get(f'p{i}n', f'Ch{i}')
            labels.append(str(pnn).strip() if pnn else f'Ch{i}')
        self.channels = labels
        self.data = pd.DataFrame(events, columns=labels)


def find_channel(columns, keywords):
    for col in columns:
        for kw in keywords:
            if kw.upper() in col.upper():
                return col
    return None


def biex(x):
    return np.arcsinh(np.asarray(x, float) / 150) * 50


def point_in_polygon(x, y, polygon):
    if polygon is None or len(polygon) < 3:
        return np.zeros(len(x), dtype=bool)
    path = Path(polygon)
    return path.contains_points(np.column_stack([x, y]))


def apply_gate(data, x_ch, y_ch, polygon, parent_mask=None):
    if x_ch is None or y_ch is None or polygon is None or len(polygon) < 3:
        return pd.Series(False, index=data.index)
    base = parent_mask.values if parent_mask is not None else np.ones(len(data), dtype=bool)
    x, y = data[x_ch].values, data[y_ch].values
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0) & base
    if not valid.any():
        return pd.Series(False, index=data.index)
    in_poly = point_in_polygon(biex(x), biex(y), polygon)
    result = np.zeros(len(data), dtype=bool)
    result[valid & in_poly] = True
    return pd.Series(result, index=data.index)


def auto_gate_gmm(data, x_ch, y_ch, parent_mask=None, n_comp=2, mode='main'):
    if x_ch is None or y_ch is None:
        return None
    subset = data[parent_mask] if parent_mask is not None and parent_mask.sum() > 0 else data
    if len(subset) < 100:
        return None
    x, y = subset[x_ch].values, subset[y_ch].values
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if valid.sum() < 100:
        return None
    xt, yt = biex(x[valid]), biex(y[valid])
    X = np.column_stack([xt, yt])
    try:
        gmm = GaussianMixture(n_components=n_comp, covariance_type='full', random_state=42, n_init=3)
        gmm.fit(X)
        labels = gmm.predict(X)
        if mode == 'main':
            target = np.argmax([np.sum(labels == i) for i in range(n_comp)])
        elif mode == 'low_x':
            target = np.argmin([np.mean(xt[labels == i]) for i in range(n_comp)])
        elif mode == 'high_x':
            target = np.argmax([np.mean(xt[labels == i]) for i in range(n_comp)])
        elif mode == 'high_x_low_y':
            target = np.argmax([np.mean(xt[labels == i]) - np.mean(yt[labels == i]) for i in range(n_comp)])
        else:
            target = 0
        mask = labels == target
        cx, cy = xt[mask], yt[mask]
        if len(cx) < 30:
            return None
        pts = np.column_stack([cx, cy])
        hull = ConvexHull(pts)
        hp = pts[hull.vertices]
        center = hp.mean(axis=0)
        polygon = [(center[0] + 1.1 * (p[0] - center[0]), center[1] + 1.1 * (p[1] - center[1])) for p in hp]
        if len(polygon) > 12:
            polygon = polygon[::max(1, len(polygon)//10)]
        return polygon
    except:
        return None


def apply_learned_adj(polygon, gate_name):
    if polygon is None:
        return None
    params = load_learned_params()
    if gate_name in params['gates']:
        adj = params['gates'][gate_name]['avg_adjustment']
        if abs(adj['x']) > 0.5 or abs(adj['y']) > 0.5:
            return [(p[0] + adj['x'], p[1] + adj['y']) for p in polygon]
    return polygon


def create_plot(data, x_ch, y_ch, title, polygon, parent_mask, gate_name):
    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=100)
    if x_ch is None or y_ch is None:
        ax.text(0.5, 0.5, "Canal non trouvé", ha='center', va='center', transform=ax.transAxes)
        return fig, 0, 0
    subset = data[parent_mask] if parent_mask is not None and parent_mask.sum() > 0 else data
    n_parent = len(subset)
    if n_parent == 0:
        ax.text(0.5, 0.5, "Pas de données", ha='center', va='center', transform=ax.transAxes)
        return fig, 0, 0
    x, y = subset[x_ch].values, subset[y_ch].values
    valid = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    xt, yt = biex(x[valid]), biex(y[valid])
    if len(xt) > 8000:
        idx = np.random.choice(len(xt), 8000, replace=False)
        xt, yt = xt[idx], yt[idx]
    ax.scatter(xt, yt, s=1, c=yt, cmap='viridis', alpha=0.5, rasterized=True)
    n_in, pct = 0, 0
    if polygon and len(polygon) >= 3:
        poly_patch = patches.Polygon(polygon, fill=True, facecolor='red', edgecolor='red', alpha=0.15, linewidth=2)
        ax.add_patch(poly_patch)
        px = [p[0] for p in polygon] + [polygon[0][0]]
        py = [p[1] for p in polygon] + [polygon[0][1]]
        ax.plot(px, py, 'r-', linewidth=2)
        full_mask = apply_gate(data, x_ch, y_ch, polygon, parent_mask)
        n_in = full_mask.sum()
        pct = n_in / n_parent * 100 if n_parent > 0 else 0
        cx, cy = np.mean(px[:-1]), np.mean(py[:-1])
        ax.annotate(f"{gate_name}\n{pct:.1f}%\n({n_in:,})", (cx, cy), ha='center', va='center',
                   fontsize=9, fontweight='bold', color='darkred',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xlabel(x_ch, fontweight='bold')
    ax.set_ylabel(y_ch, fontweight='bold')
    ax.set_title(f"{title}\n(n={n_parent:,})", fontsize=10, fontweight='bold')
    ax.set_facecolor('white')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig, n_in, pct


def move_polygon(polygon, dx, dy):
    return [(p[0] + dx, p[1] + dy) for p in polygon] if polygon else None


def scale_polygon(polygon, factor):
    if not polygon:
        return None
    center = np.mean(polygon, axis=0)
    return [(center[0] + factor * (p[0] - center[0]), center[1] + factor * (p[1] - center[1])) for p in polygon]


# ===== MAIN =====
st.markdown('<h1 class="main-header">🔬 FACS Auto-Gating</h1>', unsafe_allow_html=True)

learned = load_learned_params()
n_learned = learned.get('n_corrections', 0)
if n_learned > 0:
    st.info(f"🧠 {n_learned} correction(s) apprises")

for key in ['reader', 'data', 'channels', 'polygons', 'original_polygons', 'auto_done']:
    if key not in st.session_state:
        st.session_state[key] = {} if 'polygon' in key or key == 'channels' else None
if st.session_state.get('auto_done') is None:
    st.session_state.auto_done = False

uploaded = st.file_uploader("📁 Fichier FCS", type=['fcs'])

if uploaded:
    if st.session_state.reader is None or st.session_state.get('fname') != uploaded.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.fcs') as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        with st.spinner("Chargement..."):
            reader = FCSReader(tmp_path)
            st.session_state.reader = reader
            st.session_state.data = reader.data
            st.session_state.fname = uploaded.name
            st.session_state.polygons = {}
            st.session_state.original_polygons = {}
            st.session_state.auto_done = False
            cols = reader.data.columns
            st.session_state.channels = {
                'FSC-A': find_channel(cols, ['FSC-A', 'FSC']),
                'FSC-H': find_channel(cols, ['FSC-H']),
                'SSC-A': find_channel(cols, ['SSC-A', 'SSC']),
                'LiveDead': find_channel(cols, ['LiveDead', 'Viab', 'Aqua', 'Live']),
                'hCD45': find_channel(cols, ['PerCP', 'CD45']),
                'CD3': find_channel(cols, ['AF488', 'FITC', 'CD3']),
                'CD19': find_channel(cols, ['PE-Fire', 'CD19']),
                'CD4': find_channel(cols, ['BV650', 'CD4']),
                'CD8': find_channel(cols, ['BUV805', 'APC-Cy7', 'CD8']),
            }

    reader = st.session_state.reader
    data = st.session_state.data
    ch = st.session_state.channels
    n_total = len(data)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Événements", f"{n_total:,}")
    c2.metric("Canaux", len(reader.channels))
    c3.metric("Fichier", reader.filename[:25])
    
    with st.expander("📋 Canaux détectés"):
        for name, canal in ch.items():
            st.write(f"{'✅' if canal else '❌'} **{name}**: {canal or 'Non trouvé'}")
        st.write("**Tous:**", ", ".join(reader.channels))
    
    if ch['FSC-A'] is None or ch['SSC-A'] is None:
        st.error("❌ FSC-A ou SSC-A non trouvé!")
        st.stop()
    
    st.markdown("---")
    
    if not st.session_state.auto_done:
        if st.button("🚀 LANCER L'AUTO-GATING", type="primary", use_container_width=True):
            prog = st.progress(0)
            
            # Cells
            poly = auto_gate_gmm(data, ch['FSC-A'], ch['SSC-A'], None, 2, 'main')
            poly = apply_learned_adj(poly, 'cells')
            st.session_state.polygons['cells'] = poly
            st.session_state.original_polygons['cells'] = list(poly) if poly else None
            prog.progress(20)
            
            # Singlets
            if ch['FSC-H']:
                cells_m = apply_gate(data, ch['FSC-A'], ch['SSC-A'], poly, None)
                poly = auto_gate_gmm(data, ch['FSC-A'], ch['FSC-H'], cells_m, 2, 'main')
                poly = apply_learned_adj(poly, 'singlets')
            else:
                poly = None
            st.session_state.polygons['singlets'] = poly
            st.session_state.original_polygons['singlets'] = list(poly) if poly else None
            prog.progress(40)
            
            # Live
            if ch['LiveDead']:
                cells_m = apply_gate(data, ch['FSC-A'], ch['SSC-A'], st.session_state.polygons['cells'], None)
                if st.session_state.polygons['singlets']:
                    sing_m = apply_gate(data, ch['FSC-A'], ch['FSC-H'], st.session_state.polygons['singlets'], cells_m)
                else:
                    sing_m = cells_m
                poly = auto_gate_gmm(data, ch['LiveDead'], ch['SSC-A'], sing_m, 2, 'low_x')
                poly = apply_learned_adj(poly, 'live')
            else:
                poly = None
            st.session_state.polygons['live'] = poly
            st.session_state.original_polygons['live'] = list(poly) if poly else None
            prog.progress(60)
            
            # hCD45
            if ch['hCD45']:
                cells_m = apply_gate(data, ch['FSC-A'], ch['SSC-A'], st.session_state.polygons['cells'], None)
                sing_m = apply_gate(data, ch['FSC-A'], ch['FSC-H'], st.session_state.polygons['singlets'], cells_m) if st.session_state.polygons['singlets'] else cells_m
                live_m = apply_gate(data, ch['LiveDead'], ch['SSC-A'], st.session_state.polygons['live'], sing_m) if st.session_state.polygons['live'] else sing_m
                poly = auto_gate_gmm(data, ch['hCD45'], ch['SSC-A'], live_m, 2, 'high_x')
                poly = apply_learned_adj(poly, 'hcd45')
            else:
                poly = None
            st.session_state.polygons['hcd45'] = poly
            st.session_state.original_polygons['hcd45'] = list(poly) if poly else None
            prog.progress(80)
            
            # T cells et CD4
            st.session_state.polygons['t_cells'] = None
            st.session_state.polygons['cd4'] = None
            st.session_state.original_polygons['t_cells'] = None
            st.session_state.original_polygons['cd4'] = None
            prog.progress(100)
            
            st.session_state.auto_done = True
            st.rerun()
    
    if st.session_state.auto_done:
        polygons = st.session_state.polygons
        
        # Recalcul masques
        cells_m = apply_gate(data, ch['FSC-A'], ch['SSC-A'], polygons.get('cells'), None)
        sing_m = apply_gate(data, ch['FSC-A'], ch['FSC-H'], polygons.get('singlets'), cells_m) if polygons.get('singlets') else cells_m
        live_m = apply_gate(data, ch['LiveDead'], ch['SSC-A'], polygons.get('live'), sing_m) if polygons.get('live') else sing_m
        hcd45_m = apply_gate(data, ch['hCD45'], ch['SSC-A'], polygons.get('hcd45'), live_m) if polygons.get('hcd45') else live_m
        
        stats = []
        
        # CELLS
        st.markdown("### 1️⃣ Cells")
        col1, col2 = st.columns([2, 1])
        with col1:
            fig, n, p = create_plot(data, ch['FSC-A'], ch['SSC-A'], 'Ungated → Cells', polygons.get('cells'), None, 'Cells')
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with col2:
            st.markdown(f"**Cells:** {n:,} ({p:.1f}%)")
            c1, c2 = st.columns(2)
            if c1.button("⬆️", key="c_up"): st.session_state.polygons['cells'] = move_polygon(polygons['cells'], 0, 5); st.rerun()
            if c1.button("⬇️", key="c_dn"): st.session_state.polygons['cells'] = move_polygon(polygons['cells'], 0, -5); st.rerun()
            if c2.button("⬅️", key="c_lt"): st.session_state.polygons['cells'] = move_polygon(polygons['cells'], -5, 0); st.rerun()
            if c2.button("➡️", key="c_rt"): st.session_state.polygons['cells'] = move_polygon(polygons['cells'], 5, 0); st.rerun()
            if c1.button("➕", key="c_gr"): st.session_state.polygons['cells'] = scale_polygon(polygons['cells'], 1.1); st.rerun()
            if c2.button("➖", key="c_sh"): st.session_state.polygons['cells'] = scale_polygon(polygons['cells'], 0.9); st.rerun()
        stats.append(('Cells', 'Ungated', n, p))
        
        # SINGLETS
        if polygons.get('singlets'):
            st.markdown("### 2️⃣ Singlets")
            col1, col2 = st.columns([2, 1])
            with col1:
                fig, n, p = create_plot(data, ch['FSC-A'], ch['FSC-H'], 'Cells → Singlets', polygons.get('singlets'), cells_m, 'Singlets')
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            with col2:
                st.markdown(f"**Singlets:** {n:,} ({p:.1f}%)")
                c1, c2 = st.columns(2)
                if c1.button("⬆️", key="s_up"): st.session_state.polygons['singlets'] = move_polygon(polygons['singlets'], 0, 5); st.rerun()
                if c1.button("⬇️", key="s_dn"): st.session_state.polygons['singlets'] = move_polygon(polygons['singlets'], 0, -5); st.rerun()
                if c2.button("⬅️", key="s_lt"): st.session_state.polygons['singlets'] = move_polygon(polygons['singlets'], -5, 0); st.rerun()
                if c2.button("➡️", key="s_rt"): st.session_state.polygons['singlets'] = move_polygon(polygons['singlets'], 5, 0); st.rerun()
                if c1.button("➕", key="s_gr"): st.session_state.polygons['singlets'] = scale_polygon(polygons['singlets'], 1.1); st.rerun()
                if c2.button("➖", key="s_sh"): st.session_state.polygons['singlets'] = scale_polygon(polygons['singlets'], 0.9); st.rerun()
            stats.append(('Singlets', 'Cells', n, p))
        
        # LIVE
        if polygons.get('live'):
            st.markdown("### 3️⃣ Live")
            col1, col2 = st.columns([2, 1])
            with col1:
                fig, n, p = create_plot(data, ch['LiveDead'], ch['SSC-A'], 'Singlets → Live', polygons.get('live'), sing_m, 'Live')
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            with col2:
                st.markdown(f"**Live:** {n:,} ({p:.1f}%)")
                c1, c2 = st.columns(2)
                if c1.button("⬆️", key="l_up"): st.session_state.polygons['live'] = move_polygon(polygons['live'], 0, 5); st.rerun()
                if c1.button("⬇️", key="l_dn"): st.session_state.polygons['live'] = move_polygon(polygons['live'], 0, -5); st.rerun()
                if c2.button("⬅️", key="l_lt"): st.session_state.polygons['live'] = move_polygon(polygons['live'], -5, 0); st.rerun()
                if c2.button("➡️", key="l_rt"): st.session_state.polygons['live'] = move_polygon(polygons['live'], 5, 0); st.rerun()
                if c1.button("➕", key="l_gr"): st.session_state.polygons['live'] = scale_polygon(polygons['live'], 1.1); st.rerun()
                if c2.button("➖", key="l_sh"): st.session_state.polygons['live'] = scale_polygon(polygons['live'], 0.9); st.rerun()
            stats.append(('Live', 'Singlets', n, p))
        
        # hCD45
        if polygons.get('hcd45'):
            st.markdown("### 4️⃣ hCD45+")
            col1, col2 = st.columns([2, 1])
            with col1:
                fig, n, p = create_plot(data, ch['hCD45'], ch['SSC-A'], 'Live → hCD45+', polygons.get('hcd45'), live_m, 'hCD45+')
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            with col2:
                st.markdown(f"**hCD45+:** {n:,} ({p:.1f}%)")
                c1, c2 = st.columns(2)
                if c1.button("⬆️", key="h_up"): st.session_state.polygons['hcd45'] = move_polygon(polygons['hcd45'], 0, 5); st.rerun()
                if c1.button("⬇️", key="h_dn"): st.session_state.polygons['hcd45'] = move_polygon(polygons['hcd45'], 0, -5); st.rerun()
                if c2.button("⬅️", key="h_lt"): st.session_state.polygons['hcd45'] = move_polygon(polygons['hcd45'], -5, 0); st.rerun()
                if c2.button("➡️", key="h_rt"): st.session_state.polygons['hcd45'] = move_polygon(polygons['hcd45'], 5, 0); st.rerun()
                if c1.button("➕", key="h_gr"): st.session_state.polygons['hcd45'] = scale_polygon(polygons['hcd45'], 1.1); st.rerun()
                if c2.button("➖", key="h_sh"): st.session_state.polygons['hcd45'] = scale_polygon(polygons['hcd45'], 0.9); st.rerun()
            stats.append(('hCD45+', 'Live', n, p))
        
        st.markdown("---")
        
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("💾 Sauvegarder corrections", type="primary", use_container_width=True):
                n_saved = 0
                for gname in polygons:
                    curr, orig = polygons.get(gname), st.session_state.original_polygons.get(gname)
                    if curr and orig and list(curr) != list(orig):
                        update_learned_params(gname, orig, curr)
                        n_saved += 1
                st.success(f"✅ {n_saved} correction(s) sauvegardée(s)!" if n_saved else "Aucune modification")
        with col_b:
            if st.button("🔃 Réinitialiser", use_container_width=True):
                st.session_state.polygons = {k: list(v) if v else None for k, v in st.session_state.original_polygons.items()}
                st.rerun()
        
        st.markdown("### 📊 Résumé")
        df = pd.DataFrame(stats, columns=['Population', 'Parent', 'Count', '% Parent'])
        df['% Total'] = (df['Count'] / n_total * 100).round(2)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        c1, c2 = st.columns(2)
        c1.download_button("📥 CSV", df.to_csv(index=False), f"{reader.filename}.csv", "text/csv", use_container_width=True)
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine='openpyxl')
        buf.seek(0)
        c2.download_button("📥 Excel", buf, f"{reader.filename}.xlsx", use_container_width=True)

else:
    st.markdown('<div class="info-box"><h3>🔬 Auto-Gating</h3><p>Uploadez un fichier FCS pour commencer.</p></div>', unsafe_allow_html=True)

st.caption(f"🔬 FACS Auto-Gating | 🧠 {n_learned} corrections apprises")
