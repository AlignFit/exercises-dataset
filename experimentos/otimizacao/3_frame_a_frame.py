import sys,numpy as np,pandas as pd,warnings;warnings.filterwarnings('ignore')
sys.path.insert(0, __import__('os').path.dirname(__file__)); from feats_exploracao import *
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler; from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression; from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
df=pd.read_csv(sys.argv[1])
d=pessoa_principal(df).sort_values(['video','frame']).reset_index(drop=True)
F=frame_feats(d).replace([np.inf,-np.inf],np.nan)
# janela temporal: média e desvio móveis (5 frames) por vídeo
W=F.groupby(d.video.values).rolling(5,min_periods=1,center=True)
Fm=W.mean().reset_index(level=0,drop=True).sort_index().add_suffix('__w')
Fs=W.std().reset_index(level=0,drop=True).sort_index().add_suffix('__ws')
FF=pd.concat([F,Fm,Fs],axis=1)
meta=d[['video','tipo_exercicio','tipo_execucao','pessoa']]
imp=lambda: SimpleImputer(strategy='median')
mods={'lr':lambda:make_pipeline(imp(),StandardScaler(),LogisticRegression(C=.05,class_weight='balanced',max_iter=3000)),
      'et':lambda:make_pipeline(imp(),ExtraTreesClassifier(300,min_samples_leaf=20,class_weight='balanced',n_jobs=-1,random_state=0))}
for e in ['agachamento','biceps','desenvolvimento','elevacao_lateral']:
    s=(meta.tipo_exercicio==e).values; X=FF[s].values; M=meta[s].reset_index(drop=True); y=(M.tipo_execucao=='errada').astype(int).values; ext=(M.pessoa=='externo').values
    out=[]
    for mn,mf in mods.items():
        for uext in (False,True):
            fb=[]
            for p in ['bruno','caio','paulo']:
                te=(M.pessoa==p).values; tr=~te&(np.ones(len(y),bool) if uext else ~ext)
                # pondera para cada vídeo contar igual
                w=1/M[tr].groupby('video').video.transform('count').values
                m=mf(); m.fit(X[tr],y[tr],**{m.steps[-1][0]+'__sample_weight':w})
                pr=pd.Series(m.predict_proba(X[te])[:,1]).groupby(M.video[te].values).mean()
                yv=pd.Series(y[te]).groupby(M.video[te].values).first().loc[pr.index]
                fb.append(balanced_accuracy_score(yv,(pr>=.5).astype(int)))
            out.append(f'{mn}{"_ext" if uext else ""}={np.mean(fb):.3f}')
    print(e,' '.join(out))
