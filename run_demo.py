"""Stage 6.1: one command shows the input, the solution and the plots.

  python run_demo.py                                   # hour 540, K=20, W=8, FC+AC-3, NN forecast
  python run_demo.py --hour 540 --K 20 --W 4 --plots   # feedback loop (requests dropped) + figures in outputs/
  python run_demo.py --nn off                          # last-hour forecast instead of the NN
  python run_demo.py --trace                           # UCS vs A* expansion trace (default pair KSCYng -> LOSAng)
  python run_demo.py --trace --src CHINng --dst LOSAng # trace for another pair
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
import numpy as np
from integration.pipeline import (default_planner, plan, evaluate, nn_forecast_fn, last_hour_forecast_fn, METHODS)

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description="NN -> Search -> CSP lightpath planner demo")
    ap.add_argument("--hour", type=int, default=540, help="target hour (test week is 504-671)")
    ap.add_argument("--K", type=int, default=20, help="number of requests (top-K predicted pairs)")
    ap.add_argument("--W", type=int, default=8, help="number of wavelengths")
    ap.add_argument("--method", default="fc+ac3", choices=sorted(METHODS), help="CSP method")
    ap.add_argument("--nn", default="on", choices=["on", "off"], help="off = last-hour forecast instead of the NN")
    ap.add_argument("--seed", type=int, default=0, help="NN seed (0-4)")
    ap.add_argument("--strict", action="store_true", help="never drop requests; report the CSP no-solution message")
    ap.add_argument("--routes", default="chosen", choices=["none", "chosen", "all"], help="how many routes to print")
    ap.add_argument("--plots", action="store_true", help="write the figures (default folder: outputs/)")
    ap.add_argument("--outdir", default=None, help="folder for the figures (default: outputs/)")
    ap.add_argument("--trace", action="store_true", help="print the UCS vs A* expansion trace")
    ap.add_argument("--src", help="trace source node (default KSCYng)"); ap.add_argument("--dst", help="trace destination node (default LOSAng)")
    a = ap.parse_args()

    planner, H, cfg = default_planner()
    if a.trace:                                                   # validate before the (long) report is printed
        known = set(planner.graph.nodes)
        t_src, t_dst = a.src or "KSCYng", a.dst or "LOSAng"
        if t_src not in known or t_dst not in known or t_src == t_dst:
            sys.exit(f"--src/--dst must be two different Abilene nodes: {sorted(known)}")
    lo, hi = cfg["split"]["test_hours"]
    if not (168 <= a.hour <= hi):
        sys.exit(f"hour must be in 168..{hi} (needs 168 h of history; test week is {lo}-{hi})")
    fn = nn_forecast_fn(a.seed) if a.nn == "on" else last_hour_forecast_fn(H)
    predicted = fn(a.hour)

    print(f"=== INPUT: hour {a.hour}, K={a.K}, W={a.W}, method={a.method}, "
          f"forecast={'NN (seed %d)' % a.seed if a.nn == 'on' else 'last hour (t-1)'}, strict={a.strict} ===")
    result = plan(planner, a.hour, predicted, a.K, a.W, method=a.method, strict=a.strict,
                  max_assignments=cfg["csp"]["max_assignments"])
    true_top = set(np.argsort(-H[a.hour], kind="stable")[:a.K].tolist())
    print(f"predicted top-{a.K} pairs (priority order); 'actual' is the real demand of that hour, shown for comparison only")
    print(f"  {'rank':>4}  {'pair':<16} {'predicted':>10} {'actual':>10}  in true top-{a.K}")
    for i, r in enumerate(result.requests, 1):
        print(f"  {i:>4}  {r.src + '->' + r.dst:<16} {r.demand:>9.1f}M {H[a.hour][r.id]:>9.1f}M  {'yes' if r.id in true_top else 'no'}")

    print(f"\n=== FEEDBACK LOOP ({len(result.iterations)} solves) ===")
    print(result.log_text())

    if result.status == "ok" and a.routes != "none":
        print("\n=== ROUTES ===")
        for lp in result.lightpaths:
            rid = lp["id"]
            cands = result.routes_by_request[rid]
            print(f"  {lp['src']}->{lp['dst']}: {len(cands)} candidate route(s) within reach")
            for ri, rt in enumerate(cands):
                chosen = lp["route"] == rt.nodes and lp["length_km"] == rt.length_km
                if a.routes == "all" or chosen:
                    print(f"     {'*' if chosen else ' '} [{ri}] {'-'.join(rt.nodes)}  {rt.length_km:.0f} km")

    print("\n=== ASSIGNMENT TABLE ===")
    if result.status == "ok":
        print(f"  {'pair':<16} {'demand':>9}  {'wavelength':>10}  {'km':>6}  route")
        for lp in result.lightpaths:
            print(f"  {lp['src'] + '->' + lp['dst']:<16} {lp['demand']:>8.1f}M  {lp['wavelength']:>10}  {lp['length_km']:>6.0f}  {'-'.join(lp['route'])}")
    else:
        print(f"  {result.reason}   [status: {result.status}]")
    print(f"\n=== BLOCKED ({len(result.blocked)}) ===")
    for r, why in result.blocked:
        print(f"  {r.src}->{r.dst} (predicted {r.demand:.1f} Mbps): {why}")
    if not result.blocked:
        print("  none")

    m = evaluate(result, H[a.hour])
    print(f"\n=== SUMMARY === served {m['n_served']}/{len(result.requests)}  served share {m['served_share']:.3f}  "
          f"blocking rate {m['blocking_rate']:.3f}  top-K hit rate {m['topk_hit_rate']:.2f}  "
          f"CSP assignments {m['assignments']} backtracks {m['backtracks']}  solver time {m['solver_time']:.3f}s")

    if a.trace:
        from search.trace import expansion_trace, format_trace
        src, dst = a.src or "KSCYng", a.dst or "LOSAng"          # default pair: a largest A* saving on Abilene (tied)
        print(f"\n=== SEARCH TRACE (UCS vs A*), {src} -> {dst} ===")
        print(format_trace(expansion_trace(planner.graph, src, dst, "ucs"),
                           expansion_trace(planner.graph, src, dst, "astar")))

    if a.plots:
        from eval import plots
        from search.topology import load_topology
        topo = load_topology(os.path.join(ROOT, cfg["paths"]["abilene_topology"]))
        default_out = os.path.join(ROOT, cfg["paths"]["outputs"])
        out = a.outdir or default_out
        os.makedirs(out, exist_ok=True)
        tag = f"h{a.hour}_K{a.K}_W{a.W}"
        print("\n=== FIGURES ===")
        if result.status == "ok":
            print("  ", plots.plot_topology(result, topo.nodes, topo.links, os.path.join(out, f"demo_topology_{tag}.png")))
            print("  ", plots.plot_heatmap(result, topo.links, os.path.join(out, f"demo_heatmap_{tag}.png")))
        else:
            print("   (no topology/heatmap figure: strict-mode run without a solution)")
        data = plots.load_nn_effect(os.path.join(default_out, "nn_effect.json"))
        if data:
            print("  ", plots.plot_nn_effect(data, os.path.join(out, "demo_nn_effect.png")))
        else:
            print("   NN on/off figure needs: python -m eval.nn_effect --K 20 --W 4 8 --seeds 0")
        from eval.plot_csp import make_plot, demo_panels           # Member B's plot (c): domain sizes
        print("  ", make_plot(demo_panels(a.hour), os.path.join(out, f"demo_csp_domains_h{a.hour}.png")))


if __name__ == "__main__":
    main()
