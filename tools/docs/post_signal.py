"""The figure of mallard post signal: the spectrum of the wake probe of examples/cylinder (Re = 100), from its
probe file.

    python tools/docs/post_signal.py --mallard build/src/mallard --probe RUN/solut/probe_wake_point.csv

RUN is a run of examples/cylinder (its mesh made as its input says, to t = 160). The probe file's U_Y
from t = 60 and what mallard post signal --format json reports of it go to post_signal.json, which
make_figures.py draws without running anything.
"""
import argparse
import csv
import json
import subprocess
from pathlib import Path

import numpy as np

import figstyle

DATA = Path(__file__).resolve().parent / "post_signal.json"
OPTIONS = ["--column", "U_Y", "--times", "60:", "--strouhal", "1,0.2", "--format", "json"]


def measure(mallard, probe):
    report = json.loads(subprocess.run([mallard, "post", "signal", probe, *OPTIONS], check=True,
                                       capture_output=True, text=True).stdout)
    t, x = [], []
    with open(probe) as f:
        rows = csv.reader(line for line in f if not line.startswith("#"))
        header = next(rows)
        it, ix = header.index("t"), header.index("U_Y")
        for row in rows:
            if float(row[it]) >= 60.0:
                t.append(float(row[it]))
                x.append(float(row[ix]))
    report.pop("file")
    return {"command": "mallard post signal solut/probe_wake_point.csv " + " ".join(OPTIONS[:-2]),
            "report": report, "t": t, "U_Y": x}


def figure(data):
    r = data["report"]
    sp, st = r["spectrum"], r["statistics"]
    t, x = np.array(data["t"]), np.array(data["U_Y"])
    St, psd = np.array(sp["St"]), np.array(sp["psd"])

    def draw(P):
        fig, (ax0, ax1) = figstyle.plt.subplots(1, 2, figsize=(11.5, 4.0), gridspec_kw={"width_ratios": [1.25, 1]})
        ax0.plot(t, x, color=P["teal"], lw=1.2)
        if st["stationary"]:
            ax0.axvspan(t[0], st["t_stationary"], color=P["faint"], lw=0)
        mean, ci = st["mean"], st["mean_ci"]
        ax0.axhline(mean, color=P["muted"], lw=0.8, ls="--")
        ax0.set_xlabel("t")
        ax0.set_ylabel("U_Y at x = 2.5")
        ax0.set_xlim(t[0], t[-1])
        ax0.set_title(f"mean {mean:+.4f} ± {ci:.4f}, RMS {st['rms']:.4f}", loc="left", color=P["muted"],
                      fontsize=12)

        keep = St > 0
        ax1.semilogy(St[keep], psd[keep], color=P["teal"], lw=1.4, marker="o", ms=3.5)
        peak = sp["peak"]
        ax1.axvline(peak["St"], color=P["flame"], lw=1.0, ls="--")
        figstyle.label(ax1, peak["St"] + 0.012, peak["psd"] * 0.6,
                       f"St = {peak['St']:.4f}\namplitude {peak['amplitude']:.3f}", P, color=P["flame"],
                       size=12, ha="left", va="top")
        ax1.set_xlim(0, 0.6)
        ax1.set_ylim(psd[keep].max() * 1e-7, psd[keep].max() * 5)
        ax1.set_xlabel("St = f D / U")
        ax1.set_ylabel("PSD of U_Y")
        ax1.set_title(f"Welch, {sp['window']} window, bin width {St[1] - St[0]:.3f} in St", loc="left",
                      color=P["muted"], fontsize=12)
        fig.tight_layout()
        return fig

    figstyle.save("post_signal", draw)


def figures():
    """The figure from the saved analysis (make_figures.py)."""
    figure(json.loads(DATA.read_text()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mallard", help="Mallard executable (without it, the saved analysis is redrawn)")
    ap.add_argument("--probe", help="probe_wake_point.csv of a run of examples/cylinder")
    args = ap.parse_args()
    if args.mallard:
        if not args.probe:
            ap.error("--mallard needs --probe")
        DATA.write_text(json.dumps(measure(args.mallard, args.probe), indent=1) + "\n")
    figures()


if __name__ == "__main__":
    main()
