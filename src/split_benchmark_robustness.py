"""The site-1 split benchmark under other learners and feature sets.

The reported value must not depend on the choice of proxy, so the same protocol is run
with a deeper gradient-boosted fit, with a random forest, and with the diffused ratio
removed from the features.

    python3 -m src.split_benchmark_robustness
"""
import numpy as np, json
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from .split_power_site1 import _load_bucheon
def splits(net,c):
    t=defaultdict(float)
    for i,m in enumerate(net.movements): t[m.from_edge]+=c[i]
    return np.array([c[i]/t[net.movements[i].from_edge] if t[net.movements[i].from_edge]>0 else 0.0
                     for i in range(len(net.movements))],float)

def run():

    D=_load_bucheon(60); net,Yb,S=D["net"],D["Yb"],D["S"]
    windows=json.load(open("data/bucheon_cam_turn_60min.json"))["windows"]
    clock=np.array([w[8:10] for w in windows])
    Sp=np.vstack([splits(net,Yb[w]) for w in range(S)])
    NAIVE=np.zeros_like(Sp)
    for w in range(S):
        p=[u for u in range(S) if u!=w and clock[u]==clock[w]] or [u for u in range(S) if u!=w]
        NAIVE[w]=Sp[p].mean(0)
    def bench(est_fn, use_hour=True, use_ratio=True):
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
                        f=[D["turn"][m],D["lanes"][m],D["dm"][m],NAIVE[w][m],D["appr_hist"][m]]
                        if use_hour: f.append(D["hour"][w])
                        if use_ratio: f += [r[m], D["appr_hist"][m]*r[m]]
                        X.append(f); y.append(Sp[w][m]); n.append(NAIVE[w][m])
                return np.array(X,float),np.array(y,float),np.array(n,float)
            Xtr,ytr,_=rows(oth); Xte,yte,nte=rows(q)
            P.append(est_fn().fit(Xtr,ytr).predict(Xte)); YY.append(yte); NN.append(nte)
        P=np.concatenate(P);YY=np.concatenate(YY);NN=np.concatenate(NN)
        return float(1-np.sqrt(((P-YY)**2).mean())/np.sqrt(((NN-YY)**2).mean()))
    print("  site 1 @60min, 학습기·특징 변형에 대한 견고성")
    print(f"    GBM 300 iters, 전체 특징      : {bench(lambda: HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05)):+.4f}")
    print(f"    GBM 1000 iters               : {bench(lambda: HistGradientBoostingRegressor(max_iter=1000,learning_rate=0.05)):+.4f}")
    print(f"    RandomForest 400             : {bench(lambda: RandomForestRegressor(n_estimators=400,random_state=0,n_jobs=8)):+.4f}")
    print(f"    GBM, 확산비 제거              : {bench(lambda: HistGradientBoostingRegressor(max_iter=300,learning_rate=0.05),use_ratio=False):+.4f}")



if __name__ == "__main__":
    run()
