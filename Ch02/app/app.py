"""A minimal decentralized recommender, served from the Chapter 2 code.

This is not a benchmark viewer. It is the recommender the chapter builds, running
on simulated decentralized hardware, answering one query at a time so you can see
what the two methods actually recommend and what each one cost to assemble.

The comparison is the chapter's: blocked matrix multiplication from section 2.2
against DeMM from section 2.3, both computing the same item-item model.

    blocked   cuts the OUTPUT. worker (I,J) gets the item columns of R for blocks
              I and J, both spanning every user, and returns one exact tile. the
              tiles are disjoint, so one copy of the model comes back.
    DeMM      cuts the CONTRACTION. worker r gets its own users and nothing else,
              and returns thin factors instead of its full-size partial product.

Both directions are counted, because they tell different halves of the story.
Blocked has to broadcast each item block to every worker in its grid row and
column, so its outbound cost is several copies of the dataset. DeMM sends each
user's records to exactly one worker.

Everything here is layout. The recommender is the chapter's own modules, imported
unchanged, so the app cannot drift from the book.

    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import blocked_matmul_coordinator as blocked
import demm_gram_coordinator as demm_gram
import matmul_launcher
import movielens

LINK_MB_S = 12.5          # a modeled 100 Mbps cross-domain link
MB = 1e6

st.set_page_config(page_title="Decentralized recommender (Chapter 2)",
                   layout="wide", page_icon=":movie_camera:")


# ----------------------------------------------------------------- data + models

@st.cache_data(show_spinner="Loading MovieLens ...")
def load_data(dataset, top_items):
    return movielens.load(dataset, top_items=top_items)


@st.cache_resource(show_spinner="Building the exact model with blocked matmul ...")
def build_blocked(dataset, top_items, n_workers, participants):
    R, _, _ = load_data(dataset, top_items)
    t = time.perf_counter()
    G = blocked.blocked_gram(R, device_ids=[0] * n_workers,
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
                              help="Blocked matmul arranges these on a grid over the "
                                   "model. DeMM gives each one a block of users.")
q = st.sidebar.select_slider("DeMM rank q", options=[8, 16, 32, 64, 128], value=32,
                             help="The single knob: a smaller rank moves fewer "
                                  "bytes at a small quality cost.")

st.sidebar.markdown("**Who is reporting?**")
st.sidebar.caption(
    "Uncheck a worker to drop it. The two methods fail differently, which is the "
    "point of being able to do this. DeMM merges by addition, so a missing worker "
    "removes only its own term. Blocked matmul assembles disjoint tiles, so a "
    "missing worker leaves a hole in the model.")

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
        "modeled link speed, played back as elapsed time.")


# --------------------------------------------------------------------------- body

R, titles, info = load_data(dataset, top_items)
n_items = info["n_items"]
n_live = len(participants)

st.title("A decentralized recommender")
grid = blocked.choose_worker_grid(n_workers, n_items, n_items)
st.markdown(
    f"`{info['dataset']}` &nbsp;|&nbsp; **{info['n_users']:,}** users &nbsp;|&nbsp; "
    f"**{n_items:,}** items &nbsp;|&nbsp; **{n_live} of {n_workers}** workers reporting "
    f"&nbsp;|&nbsp; blocked grid **{grid[0]} x {grid[1]}**")

popular = np.argsort(-info["popularity"])[:400]
liked_titles = st.multiselect(
    "Pick a few films you liked", options=[titles[i] for i in popular],
    default=[titles[i] for i in popular[:3]],
    help="This builds a taste profile. The recommender scores every item against "
         "it using the item-item model the workers assembled.")

liked = [titles.index(t) for t in liked_titles]
if not liked:
    st.info("Pick at least one film to get recommendations.")
    st.stop()

G, blocked_build_s = build_blocked(dataset, top_items, n_workers, participants)
model, demm_build_s = build_demm(dataset, top_items, n_workers, q, participants)

# ------------------------------------------------------------------ byte accounting
bl_bytes = blocked.gram_bytes(n_items, info["n_interactions"], n_workers)
raw_bytes = info["n_interactions"] * 2 * 4
dm_out = raw_bytes                                  # each user's records go to one worker
dm_back = demm_gram.inbound_bytes(n_items, q, n_live)
dm_total = dm_out + dm_back

t = time.perf_counter()
top_blocked, _ = blocked.recommend(G, liked, top_n=10)
serve_blocked_ms = (time.perf_counter() - t) * 1e3

t = time.perf_counter()
top_demm, _ = demm_gram.recommend(model, liked, top_n=10)
serve_demm_ms = (time.perf_counter() - t) * 1e3

overlap = len(set(top_blocked.tolist()) & set(top_demm.tolist()))


def _mb(b):
    """Format bytes as MB with enough precision to stay honest on small datasets."""
    v = b / MB
    return f"{v:,.0f} MB" if v >= 100 else (f"{v:,.1f} MB" if v >= 10 else f"{v:,.2f} MB")


def render_blocked(c):
    c.subheader("Blocked matmul, section 2.2")
    c.caption("Every worker owns one tile of the model and returns it exactly. The "
              "tiles are disjoint, so one copy of the model comes back.")
    a, b = c.columns(2)
    a.metric("Bytes out", _mb(bl_bytes["outbound"]),
             help=f"Each item block is broadcast to the {grid[0]} + {grid[1]} workers "
                  f"in its grid row and column.")
    b.metric("Bytes back", _mb(bl_bytes["inbound"]),
             help="Exactly one copy of the model, at any worker count.")
    c.caption(f"total **{_mb(bl_bytes['total'])}**, "
              f"{bl_bytes['total']/MB/LINK_MB_S:,.1f} s at a modeled 100 Mbps link")
    for rank, i in enumerate(top_blocked, start=1):
        c.write(f"**{rank}.** {titles[i]}")


def render_demm(c):
    c.subheader(f"DeMM, section 2.3, rank q = {q}")
    c.caption("Every worker holds its own users, never forms its partial product, "
              "and returns thin factors that describe it.")
    a, b = c.columns(2)
    a.metric("Bytes out", _mb(dm_out),
             delta=f"{bl_bytes['outbound']/dm_out:,.1f}x fewer",
             help="Each user's records go to exactly one worker. No broadcast.")
    b.metric("Bytes back", _mb(dm_back),
             delta=f"{bl_bytes['inbound']/dm_back:,.0f}x fewer")
    c.caption(f"total **{_mb(dm_total)}**, "
              f"{dm_total/MB/LINK_MB_S:,.1f} s  ({bl_bytes['total']/dm_total:,.1f}x less than blocked)")
    for rank, i in enumerate(top_demm, start=1):
        same = "" if i in set(top_blocked.tolist()) else "  :orange[new]"
        c.write(f"**{rank}.** {titles[i]}{same}")


# ------------------------------------------------------------------ assemble replay

MAX_REPLAY_S = 8.0
TICK_S = 0.05
bl_eta = bl_bytes["total"] / MB / LINK_MB_S
dm_eta = dm_total / MB / LINK_MB_S


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
    """Reveal each panel when its delay elapses. The bar carries no clock: the wait
    itself is the comparison, and the assembled panel reports the modeled time."""
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
d_bl, d_dm = replay_delays(bl_eta, dm_eta)
play([
    (d_dm, f"DeMM, section 2.3, rank q = {q}", render_demm, right.empty()),
    (d_bl, "Blocked matmul, section 2.2", render_blocked, left.empty()),
])

st.divider()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Top-10 overlap", f"{overlap}/10",
          help="How many of the exact model's ten recommendations DeMM also returns.")
c2.metric("Total bytes saved", _mb(bl_bytes["total"] - dm_total))
c3.metric("Serving latency", f"{serve_demm_ms:.1f} ms",
          help="Scoring goes straight through the factors, so the full item-item "
               "matrix is never built.")
c4.metric("Model build", f"{demm_build_s:.1f} s",
          help="One-off, across all reporting workers. Cached until you change the "
               "configuration.")

if n_live < n_workers:
    holes = 100 * float(np.count_nonzero(G == 0)) / G.size
    st.warning(
        f"**{n_workers - n_live} worker(s) dropped, and the two methods degrade "
        f"differently.** Blocked matmul assembles disjoint tiles, so the missing "
        f"workers leave {holes:.0f} percent of the model as zeros, a hole rather than "
        f"a smaller answer. DeMM merges by addition, so the missing workers remove "
        f"only their own users' contributions and the model stays complete over "
        f"whoever did report. Chapter 3 is about exactly this asymmetry.")

with st.expander("What each worker sent, and in which direction"):
    st.caption(
        "The exchange is what this chapter is about. Notice that the blocked method's "
        "outbound cost grows with the grid, because every item block has to reach each "
        "worker in its row and its column, while DeMM sends each user's records once "
        "no matter how many workers there are. Notice also that neither return column "
        "depends on how many users a worker holds.")
    st.dataframe([{
        "method": "blocked matmul",
        "grid": f"{grid[0]} x {grid[1]}",
        "out (MB)": round(bl_bytes["outbound"] / MB, 2),
        "back (MB)": round(bl_bytes["inbound"] / MB, 2),
        "total (MB)": round(bl_bytes["total"] / MB, 2),
    }, {
        "method": f"DeMM rank {q}",
        "grid": f"{n_workers} user shards",
        "out (MB)": round(dm_out / MB, 2),
        "back (MB)": round(dm_back / MB, 2),
        "total (MB)": round(dm_total / MB, 2),
    }], hide_index=True, width="stretch")

st.caption(
    "Chapter 2 of Decentralized Parallel Programming. The blocked path is "
    "`blocked_matmul_coordinator.py`, the DeMM path is `demm_gram_coordinator.py` "
    "and `demm_launcher.py`, and both drive the same `MatrixMulKernel` in "
    "`matmulkernel.cu`. Run `experiment_harness.ipynb` for the measured results the "
    "book reports.")
