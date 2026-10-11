import csv, glob, sys, statistics as st, collections
rows = []
for f in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else 'results/results_wfcommons.csv')):
    try: rows += list(csv.DictReader(open(f)))
    except Exception: pass
rows = [r for r in rows if r.get('optimal')]
print("requests:", len(rows), "traces:", len({r['trace'] for r in rows}), "all optimal:", all(r['status']=='OPTIMAL' for r in rows))
algs = ['cascade', 'greedy', 'recompute_all', 'da1', 'da2', 'da3']
def gap(r, a): o = float(r['optimal']); return 0.0 if o == 0 else 100 * (float(r[a]) - o) / o
for rho in sorted({r['rho'] for r in rows}):
    rs = [r for r in rows if r['rho'] == rho]
    print(f"\nrho={rho} (N={len(rs)}): mean gap % [max] | #optimal")
    for a in algs:
        g = [gap(r, a) for r in rs]
        print(f"  {a:14s} {st.mean(g):9.1f} [{max(g):9.1f}]  opt {sum(x < 1e-6 for x in g)}/{len(g)}")
    print("  retention_opt %.1f%%  ilp max %.2fs  affected %s..%s" % (100*st.mean(float(r['retention_opt']) for r in rs),
          max(float(r['ilp_time']) for r in rs), min(int(r['affected']) for r in rs), max(int(r['affected']) for r in rs)))
by = collections.defaultdict(list)
for r in rows: by[(r['workflow'], r['rho'])].append(r)
print("\nper workflow (rho=0.1): N, cascade/opt ratio, greedy gap, DA1 gap, DA3 gap, rec_all gap")
for (w, rho), rs in sorted(by.items()):
    if rho != '0.1': continue
    print(f"  {w:12s} {len(rs):3d}  x{st.mean(float(r['cascade'])/max(1e-9,float(r['optimal'])) for r in rs):6.1f}  {st.mean(gap(r,'greedy') for r in rs):6.1f}%  {st.mean(gap(r,'da1') for r in rs):6.1f}%  {st.mean(gap(r,'da3') for r in rs):6.1f}%  {st.mean(gap(r,'recompute_all') for r in rs):6.1f}%")
