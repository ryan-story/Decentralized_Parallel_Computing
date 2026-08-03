"""A minimal decentralized recommender, served from the Chapter 2 code.

This is not a benchmark viewer. It is the recommender the chapter builds, running
on simulated decentralized hardware, answering one query at a time so you can see
what the two paths actually recommend and what each one cost to assemble.

Everything here is layout. The recommender is the chapter's own modules, imported
unchanged: `distributed_gram_coordinator` for the exact baseline and
`demm_gram_coordinator` for DeMM, both calling `MatrixMulKernel` through
`matmul_launcher`. Nothing is reimplemented, so the app cannot drift from the book.

    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import demm_gram_coordinator as demm_gram
import distributed_gram_coordinator as exact_gram
import matmul_launcher
import movielens

LINK_MB_S = 12.5          # a modeled 100 Mbps cross-domain link

st.set_page_config(page_title="Decentralized recommender (Chapter 2)",
                   layout="wide", page_icon=":movie_camera:")


# ----------------------------------------------------------------- data + models

@st.cache_data(show_spinner="Loading MovieLens ...")
def load_data(dataset, top_items):
    R, titles, info = movielens.load(dataset, top_items=top_items)
    return R, titles, info


@st.cache_resource(show_spinner="Building the exact model across workers ...")
def build_exact(dataset, top_items, n_workers, participants):
    R, _, _ = load_data(dataset, top_items)
    t = time.perf_counter()
    G = exact_gram.distributed_gram(R, device_ids=[0] * n_workers,
                                    participants=list(participants))
    return G, time.perf_counter() - t


@st.cache_resource(show_spinner="Building the DeMM summary across workers ...")
def build_demm(dataset, top_items, n_workers, q, participants):
    R, _, _ = load_data(dataset, top_items)
    t = time.perf_counter()
    model = demm_gram.demm_gram(R, device_ids=[0] * n_workers, q=q,
                                participants=list(participants))
    return model, time.perf_counter() - t


# ------------------------------------------------------------------------ sidebar

st.sidebar.title("The decentralized system")

dataset = st.sidebar.selectbox(
    "Dataset", ["ml-latest-small", "ml-25m"],
    help="ml-25m is the configuration the book reports. It downloads about 250 MB "
         "once and takes a few minutes to build.")
top_items = 4000 if dataset == "ml-25m" else None

n_workers = st.sidebar.slider("Workers", 2, 16, 8,
                              help="Users are the contracted dimension, so this is "
                                   "how many blocks of users the catalogue is "
                                   "assembled from.")
q = st.sidebar.select_slider("DeMM rank q", options=[8, 16, 32, 64, 128], value=32,
                             help="The single knob: a smaller rank moves fewer "
                                  "bytes at a small quality cost.")

st.sidebar.markdown("**Who is reporting?**")
st.sidebar.caption(
    "Uncheck a worker to drop it from the merge. Because the merge is a sum, its "
    "term simply disappears and everyone else's contribution is untouched. Both "
    "paths share this, since both merge by addition.")

cols = st.sidebar.columns(4)
participants = tuple(
    i for i in range(n_workers)
    if cols[i % 4].checkbox(f"{i}", value=True, key=f"w{i}"))

if not participants:
    st.sidebar.error("At least one worker has to report.")
    st.stop()

st.sidebar.divider()

replay_mode = st.sidebar.radio(
    "Assemble replay", ["Scaled", "Real time", "Off"], index=0,
    help="Both models are already built, so the panels below could appear at once. "
         "Replay holds each one back by its modeled transfer time instead, so the "
         "byte difference is something you wait through rather than read. Scaled "
         "compresses the slower path to a few seconds and moves the faster one by "
         "the same factor, which leaves the ratio between them intact.")

st.sidebar.divider()
st.sidebar.caption(
    f"**Evidence class: simulated_decentralized.** The arithmetic is real and runs "
    f"on {'the GPU through MatrixMulKernel' if matmul_launcher.gpu_available() else 'the CPU (no CUDA device found)'}. "
    f"The workers are threads on this one machine and the wide-area network is a "
    f"byte accounting, not a link. No number here rests on a real multi-node "
    f"deployment.")

if replay_mode != "Off":
    st.sidebar.caption(
        ":orange[**The wait is synthesized, not measured.**] Nothing is transferred "
        "and no link is exercised. The stall is the byte count divided by the "
        "modeled link speed, played back as elapsed time. It shows you the shape of "
        "the arithmetic in Table 2.1; it is not evidence for it.")


# --------------------------------------------------------------------------- body

R, titles, info = load_data(dataset, top_items)
n_items = info["n_items"]

st.title("A decentralized recommender")
st.markdown(
    f"`{info['dataset']}` &nbsp;|&nbsp; **{info['n_users']:,}** users split across "
    f"**{len(participants)} of {n_workers}** reporting workers &nbsp;|&nbsp; "
    f"**{n_items:,}** items &nbsp;|&nbsp; no worker's users ever leave it.")

popular = np.argsort(-info["popularity"])[:400]
default = [titles[i] for i in popular[:3]]

liked_titles = st.multiselect(
    "Pick a few films you liked", options=[titles[i] for i in popular],
    default=default,
    help="This builds a taste profile. The recommender scores every item against "
         "it using the item-item model the workers assembled.")

liked = [titles.index(t) for t in liked_titles]
if not liked:
    st.info("Pick at least one film to get recommendations.")
    st.stop()

G, exact_build_s = build_exact(dataset, top_items, n_workers, participants)
model, demm_build_s = build_demm(dataset, top_items, n_workers, q, participants)

exact_bytes = exact_gram.inbound_bytes(n_items, len(participants))
demm_bytes = demm_gram.inbound_bytes(n_items, q, len(participants))

t = time.perf_counter()
top_exact, score_exact = exact_gram.recommend(G, liked, top_n=10)
serve_exact_ms = (time.perf_counter() - t) * 1e3

t = time.perf_counter()
top_demm, score_demm = demm_gram.recommend(model, liked, top_n=10)
serve_demm_ms = (time.perf_counter() - t) * 1e3

overlap = len(set(top_exact.tolist()) & set(top_demm.tolist()))

exact_eta = exact_bytes / 1e6 / LINK_MB_S
demm_eta = demm_bytes / 1e6 / LINK_MB_S


def render_exact(c):
    c.subheader("Exact distributed Gram")
    c.caption("Every worker returns its full partial matrix. The merge reproduces "
              "the centralized model exactly.")
    a, b = c.columns(2)
    a.metric("Bytes received", f"{exact_bytes/1e6:,.1f} MB")
    b.metric("Time to assemble", f"{exact_eta:,.1f} s",
             help="At a modeled 100 Mbps cross-domain link.")
    for rank, i in enumerate(top_exact, start=1):
        c.write(f"**{rank}.** {titles[i]}")


def render_demm(c):
    c.subheader(f"DeMM Gram, q = {q}")
    c.caption("Every worker returns a rank-q factor and an exact item-count "
              "diagonal, and never forms its partial matrix at all.")
    a, b = c.columns(2)
    a.metric("Bytes received", f"{demm_bytes/1e6:,.2f} MB",
             delta=f"{exact_bytes/demm_bytes:,.0f}x fewer", delta_color="normal")
    b.metric("Time to assemble", f"{demm_eta:,.2f} s",
             delta=f"{exact_bytes/demm_bytes:,.0f}x faster", delta_color="normal")
    for rank, i in enumerate(top_demm, start=1):
        same = "" if i in set(top_exact.tolist()) else "  :orange[new]"
        c.write(f"**{rank}.** {titles[i]}{same}")


# ------------------------------------------------------------------ assemble replay
#
# Both models are already built and cached, so these panels could be drawn at once.
# Holding each one back by its modeled transfer time is the entire point of the
# chapter made physical: DeMM's summary lands while the exact partials are still
# arriving. The stall is played back from the byte model, so a reader should not
# mistake it for a measurement; the sidebar says so.

MAX_REPLAY_S = 8.0        # the slower path is compressed to this so the app stays usable
TICK_S = 0.05


def replay_delays(*etas):
    """Modeled seconds mapped to wall-clock stalls, preserving the ratio between them."""
    if replay_mode == "Off":
        return [0.0] * len(etas)
    if replay_mode == "Real time":
        return list(etas)
    slowest = max(etas)
    factor = MAX_REPLAY_S / slowest if slowest > MAX_REPLAY_S else 1.0
    return [e * factor for e in etas]


def play(panels):
    """Reveal each panel when its delay elapses, slowest last.

    `panels` is a sequence of `(delay, title, render, slot)`. Each slot shows a live
    progress bar until its delay is up, at which point the real panel replaces it.
    Everything is already computed, so a delay of zero renders immediately and the
    loop never runs.

    The bar deliberately carries no clock. The wait itself is the comparison, and a
    running number would invite the reader to treat a synthesized stall as a
    stopwatch reading. The assembled panel reports the modeled time.
    """
    bars = {}
    for delay, title, render, slot in panels:
        if delay <= 0.0:
            render(slot.container())
            continue
        c = slot.container()
        c.subheader(title)
        bars[title] = (c.progress(0.0, text="receiving ..."), delay, render, slot)

    t0 = time.perf_counter()
    while bars:
        elapsed = time.perf_counter() - t0
        for title, (bar, delay, render, slot) in list(bars.items()):
            if elapsed >= delay:
                render(slot.container())
                del bars[title]
            else:
                bar.progress(elapsed / delay, text="receiving ...")
        if bars:
            time.sleep(TICK_S)


left, right = st.columns(2)
d_exact, d_demm = replay_delays(exact_eta, demm_eta)
play([
    (d_demm, f"DeMM Gram, q = {q}", render_demm, right.empty()),
    (d_exact, "Exact distributed Gram", render_exact, left.empty()),
])

st.divider()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Top-10 overlap", f"{overlap}/10",
          help="How many of the exact model's ten recommendations DeMM also returns.")
c2.metric("Bytes saved", f"{(exact_bytes-demm_bytes)/1e6:,.1f} MB")
c3.metric("Serving latency", f"{serve_demm_ms:.1f} ms",
          help="Scoring goes straight through the factors, so the full item-item "
               "matrix is never built.")
c4.metric("Model build", f"{demm_build_s:.1f} s",
          help="One-off, across all reporting workers. Cached until you change the "
               "configuration.")

with st.expander("What each worker sent"):
    st.caption(
        "The exchange is what this chapter is about. Notice that neither column "
        "depends on how many users a worker holds: the exact partial is set by the "
        "size of the catalogue, and the DeMM summary by the rank. Adding users "
        "increases local work and changes the exchange not at all, which is why "
        "the DeMM advantage widens as the system grows.")
    rows = []
    for r in range(n_workers):
        reporting = r in participants
        rows.append({
            "worker": r,
            "reporting": "yes" if reporting else "no",
            "exact partial (MB)": (n_items * n_items * 4 / 1e6) if reporting else 0.0,
            "DeMM summary (MB)": ((n_items * q + n_items) * 4 / 1e6) if reporting else 0.0,
        })
    st.dataframe(rows, hide_index=True, width="stretch")

st.caption(
    "Chapter 2 of Decentralized Parallel Programming. The exact path is "
    "`distributed_gram_coordinator.py`, the compressed path is "
    "`demm_gram_coordinator.py` and `demm_launcher.py`, and both drive the same "
    "`MatrixMulKernel` in `matmulkernel.cu`. Run `experiment_harness.ipynb` for the "
    "measured results the book reports.")
