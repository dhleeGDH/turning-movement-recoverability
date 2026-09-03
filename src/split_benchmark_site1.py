"""Benchmark for the turn split at site 1, against the same-clock-hour historical share.

Leave-one-intersection-out: the model is fitted on the other intersections and applied to
the held-out one, so no query movement sees its own intersection in training. The naive
baseline is the clock-slot mean over the other windows, which is the real-site baseline of
the paper. Reported at the hourly resolution, where that baseline rests on a stable mean.

    python3 -m src.split_benchmark_site1
"""
import numpy as np, json
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingRegressor
from .split_power_site1 import _load_bucheon

import json as _json

def splits(net,c):
    t=defaultdict(float)
    for i,m in enumerate(net.movements): t[m.from_edge]+=c[i]
    return np.array([c[i]/t[net.movements[i].from_edge] if t[net.movements[i].from_edge]>0 else 0.0
                     for i in range(len(net.movements))],float)


def run():

    out={}
    for BIN in (60,15):
        D=_load_bucheon(BIN); net,Yb,S=D["net"],D["Yb"],D["S"]
        windows=_json.load(open(f"data/bucheon_cam_turn_{BIN}min.json"))["windows"]
        clock=np.array([w.split('_')[0][8:10]+'_'+w.split('_')[1] for w in windows])   # HH_MM
        Sp=np.vstack([splits(net,Yb[w]) for w in range(S)])
        NAIVE=np.zeros_like(Sp)
        npeer=[]
        for w in range(S):
            p=[u for u in range(S) if u!=w and clock[u]==clock[w]]
            npeer.append(len(p))
            NAIVE[w]=Sp[p].mean(0) if p else Sp[[u for u in range(S) if u!=w]].mean(0)
        P,YY,NN=[],[],[]
        for I in D["inters"]:
            q=D["inter_of"]==I
            if not q.any(): continue
            V=D["Vfold"][I]; vh=V.mean(0); oth=~q
            def rows(mask):
                X,y,n=[],[],[]
                for w in range(S):
                    r=V[w]/(vh+1e-9)
                    for m in np.nonzero(mask)[0]:
                        X.append([D["turn"][m],D["lanes"][m],D["dm"][m],D["hour"][w],
                                  NAIVE[w][m],D["appr_hist"][m],r[m],D["appr_hist"][m]*r[m]])
                        y.append(Sp[w][m]); n.append(NAIVE[w][m])
                return np.array(X,float),np.array(y,float),np.array(n,float)
            Xtr,ytr,_=rows(oth); Xte,yte,nte=rows(q)
            P.append(HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05).fit(Xtr,ytr).predict(Xte))
            YY.append(yte); NN.append(nte)
        P=np.concatenate(P);YY=np.concatenate(YY);NN=np.concatenate(NN)
        rn=np.sqrt(((NN-YY)**2).mean()); rm=np.sqrt(((P-YY)**2).mean()); ri=float(1-rm/rn)
        out[f"{BIN}min"]={"split_benchmark_RI":round(ri,4),"naive_rmse_share":round(float(rn),5),
                          "median_peers_per_clock_slot":int(np.median(npeer)),"n_rows":int(len(YY)),
                          "above_0.028":bool(ri>0.028),"above_0.018":bool(ri>0.018)}
        print(f"  site 1 @{BIN}min (clock key = HH_MM, peers/slot median {int(np.median(npeer))}): "
              f"split benchmark RI = {ri:+.4f}  naive RMSE {rn:.4f}  "
              f"floor 0.028: {'ABOVE' if ri>0.028 else 'below'}")
    json.dump(out, open('data/e7_split_benchmark_site1.json', 'w'), indent=1)



if __name__ == "__main__":
    run()
