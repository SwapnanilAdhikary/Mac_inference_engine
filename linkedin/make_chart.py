"""LinkedIn image: my engine vs vllm-metal (same-conditions A/B, Qwen3-1.7B bf16, M5 16 GB)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib import font_manager

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
MINE, VLLM = "#2a78d6", "#eb6834"
for f in ("Helvetica Neue", "Helvetica", "Arial"):
    if any(f == x.name for x in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break

# output tokens/s, concurrency 8 and 16 (bench A/B run v2: ours mbt=512 vs vllm-metal, back to back)
DATA = [("Chat", [(215.3, 205.6), (341.5, 318.5)]),
        ("Shared long prompt", [(206.0, 195.4), (305.2, 285.5)]),
        ("Long prompts", [(49.5, 49.7), (56.5, 54.5)])]

W, H, DPI = 1200, 1200, 100
fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=SURFACE)
T = lambda x, y, s, **k: fig.text(x / W, 1 - y / H, s, **k)

T(60, 60, "My from-scratch LLM engine vs vllm-metal", fontsize=32, weight="bold", color=INK, va="top")
T(60, 118, "Qwen3-1.7B · M5 MacBook, 16 GB · output tokens/s, higher is better", fontsize=16, color=INK2, va="top")
for i, (name, c) in enumerate([("My engine", MINE), ("vllm-metal", VLLM)]):
    x0 = 60 + i * 190
    fig.patches.append(FancyBboxPatch((x0 / W, 1 - 188 / H), 16 / W, 16 / H, boxstyle="round,pad=0,rounding_size=0.003",
                                      transform=fig.transFigure, color=c, figure=fig))
    T(x0 + 26, 180, name, fontsize=16, color=INK, va="center")

panel_w, gap, top, ph = 330, 45, 300, 700
for p, (title, groups) in enumerate(DATA):
    left = 60 + p * (panel_w + gap)
    ax = fig.add_axes([left / W, 1 - (top + ph) / H, panel_w / W, ph / H], facecolor=SURFACE)
    ymax = max(max(g) for g in groups) * 1.3
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(0, ymax)
    px_x = 2.2 / panel_w; px_y = ymax / ph                     # data units per pixel
    bw, gp, r = 24 * px_x, 10 * px_x, 4                        # 24px bars, 4px rounded data-end
    for gi, vals in enumerate(groups):
        for si, (v, c) in enumerate(zip(vals, (MINE, VLLM))):
            x = gi + (-(bw + gp) / 2 if si == 0 else (bw + gp) / 2) - bw / 2
            ax.add_patch(FancyBboxPatch((x, 0), bw, v, boxstyle=f"round,pad=0,rounding_size={r * px_x}",
                                        mutation_aspect=px_y / px_x, color=c, lw=0))
            ax.add_patch(Rectangle((x, 0), bw, v - r * px_y, color=c, lw=0))     # square at the baseline
            ax.text(x + bw / 2, v + 8 * px_y, f"{v:.0f}", ha="center", va="bottom", fontsize=12.5, color=INK2)
        d = (vals[0] / vals[1] - 1) * 100
        ax.text(gi, max(vals) + 50 * px_y, "tie" if abs(d) < 2 else f"{d:+.0f}%", ha="center", va="bottom",
                fontsize=20, weight="bold", color=INK if d >= 2 else INK2)
    ax.set_xticks([0, 1], ["8 users", "16 users"], fontsize=15, color=INK)
    ax.tick_params(axis="x", length=0, pad=10)
    ax.tick_params(axis="y", labelsize=12, colors=MUTED, length=0)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax.grid(axis="y", color=GRID, lw=1, ls="-"); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#c9c8c3")
    T(left, 262, title, fontsize=18, weight="bold", color=INK, va="bottom")

T(60, 1110, "github.com/SwapnanilAdhikary/Mac_inference_engine", fontsize=16, weight="bold", color=INK, va="top")

out = __file__.replace("make_chart.py", "engine_vs_vllm_metal.png")
fig.savefig(out, dpi=DPI, facecolor=SURFACE)
print(out)
