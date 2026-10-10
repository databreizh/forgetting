import sys, csv, time, random
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
from collaborative_forgetting_experiments import *
from mccf_ilp import exact_ilp
out = open('results/results_ilp_scalability.csv', 'w', newline='')
w = csv.writer(out); w.writerow(['n','family','fraction','seed','affected','status','ilp_time','optimal','greedy_cost','greedy_time'])
for n in (1000, 5000, 10000, 50000, 100000):
    for fam in ('LOCAL','SHARED','ADVERSARIAL'):
        for fr in (0.1, 0.25, 0.5):
            for seed in (1,2,3,4,5):
                g = generate_family_cfg(n, 5, fam, seed, trap_depth=1)
                f = choose_forgetting_request_by_fraction(g, fr, True, random.Random(seed + 10000))
                t = time.perf_counter(); p, st = exact_ilp(g, f, time_limit=600); ti = time.perf_counter() - t
                t = time.perf_counter(); gp = greedy_repair(g, f); tg = time.perf_counter() - t
                w.writerow([n, fam, fr, seed, len(affected_region(g, f)), st, round(ti,3), round(total_cost(g,p),4) if p else '', round(total_cost(g,gp),4), round(tg,4)]); out.flush()
                print(n, fam, fr, seed, st, round(ti,2), flush=True)
