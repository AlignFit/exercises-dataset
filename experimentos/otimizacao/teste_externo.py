import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, __import__('os').path.dirname(__file__))
import feats_exploracao as feats
from feats_exploracao import pessoa_principal, frame_feats, q10, q90, FC
from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler; from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression; from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier

BASE = __import__("os").path.join(__import__("os").path.dirname(__file__), "..", "..", "dados", "base-v2.csv")
TESTE = sys.argv[1]

def bio(df):
    d = pessoa_principal(df).sort_values(['video', 'frame']); F = frame_feats(d); F['video'] = d.video.values
    g = F.groupby('video'); out = []
    for nm, fn in [('mean','mean'),('std','std'),('min','min'),('max','max'),('p10',q10),('p90',q90)]:
        a = g.agg(fn); a.columns = [f'{c}__{nm}' for c in a.columns]; out.append(a)
    X = pd.concat(out, axis=1)
    for c in F.columns.drop('video'): X[f'{c}__amp'] = X[f'{c}__p90'] - X[f'{c}__p10']
    return X.replace([np.inf, -np.inf], np.nan)
def brutos(df, principal):
    d = pessoa_principal(df) if principal else df
    g = d.groupby('video'); return pd.concat([g[FC].mean(), g[FC].std().fillna(0), g[FC].min(), g[FC].max()], axis=1)

tr = pd.read_csv(BASE); te = pd.read_csv(TESTE)
meta = tr.groupby('video')[['tipo_exercicio','tipo_execucao','pessoa']].first()
Xb_tr, Xb_te = bio(tr), bio(te)
Xp_tr, Xp_te = brutos(tr, True).loc[Xb_tr.index], brutos(te, True).loc[Xb_te.index]
Xb_te = Xb_te[Xb_tr.columns]
M = meta.loc[Xb_tr.index]
imp = lambda: SimpleImputer(strategy='median')

# ---------- exercício: RF biomec+brutos ----------
Xe_tr = pd.concat([Xb_tr, Xp_tr], axis=1).values; Xe_te = pd.concat([Xb_te, Xp_te], axis=1).values
rf = make_pipeline(imp(), RandomForestClassifier(500, class_weight='balanced', random_state=0)).fit(Xe_tr, M.tipo_exercicio.values)
pe = rf.predict(Xe_te); pp = rf.predict_proba(Xe_te).max(1)

# ---------- execução: elevação lateral ----------
s = (M.tipo_exercicio == 'elevacao_lateral').values; Me = M[s]
y = (Me.tipo_execucao == 'errada').astype(int).values; ext = (Me.pessoa == 'externo').values
X = Xb_tr[s]
REL = ['ombro','cotovelo','pulso_alt','cotov_alt','pulso_lat','cotov_lat','tronco']
rel = [c for c in X.columns if any(c.startswith(r) for r in REL)]

res = pd.DataFrame(index=Xb_te.index)
res['exercicio_otim'] = pe; res['conf_ex'] = pp
lr = make_pipeline(imp(), StandardScaler(), LogisticRegression(C=.1, class_weight='balanced', max_iter=3000)).fit(X[~ext].values, y[~ext])
res['p_err_binario'] = lr.predict_proba(Xb_te.values)[:, 1]
rfe = make_pipeline(imp(), RandomForestClassifier(500, min_samples_leaf=2, class_weight='balanced', random_state=0)).fit(X.values, y)
res['p_err_rf_ext'] = rfe.predict_proba(Xb_te.values)[:, 1]
# conformidade: distância ao padrão correto (equipe + externos), limiar p75 das corretas da equipe
cor = (y == 0)
sc = make_pipeline(imp(), StandardScaler()).fit(X.loc[cor, rel].values)
dist = lambda A: np.sqrt((np.clip(np.abs(sc.transform(A)), 0, 6) ** 2).mean(1))
thr = np.percentile(dist(X.loc[cor & ~ext, rel].values), 75)
res['dist_conf'] = dist(Xb_te[rel].values); res['limiar'] = thr
for c, col in [('binario','p_err_binario'),('rf_ext','p_err_rf_ext')]: res[c] = np.where(res[col] >= .5, 'ERRADA', 'CORRETA')
res['conformidade'] = np.where(res.dist_conf > thr, 'ERRADA', 'CORRETA')
pd.set_option('display.width', 250)
print(res.round(2).to_string())
print('\nAcerto do exercício:', (res.exercicio_otim == 'elevacao_lateral').mean())
for c in ['binario','rf_ext','conformidade']: print(f'Corretas reconhecidas ({c}):', (res[c] == 'CORRETA').sum(), '/', len(res))
