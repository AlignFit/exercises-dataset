import sys,numpy as np,pandas as pd,warnings;warnings.filterwarnings('ignore')
from sklearn.metrics import accuracy_score,f1_score,balanced_accuracy_score,confusion_matrix
from xgboost import XGBClassifier
df0=pd.read_csv(sys.argv[1]); FC=[c for c in df0.columns if c.endswith('_x') or c.endswith('_y')]
def agg(fr):
    g=fr.groupby('video'); return pd.concat([g[FC].mean(),g[FC].std().fillna(0),g[FC].min(),g[FC].max()],axis=1)
HP={'agachamento':.1,'biceps':.03,'desenvolvimento':.1,'elevacao_lateral':.03}
for e,lr in HP.items():
    d=df0[df0.tipo_exercicio==e].copy(); X=agg(d); m=d.groupby('video')[['tipo_execucao','pessoa']].first().loc[X.index]
    y=(m.tipo_execucao=='errada').astype(int).values; Xv=X.values; ext=(m.pessoa=='externo').values
    for modo in ['sem_ext','com_ext','com_ext_peso']:
        P=np.zeros(len(y),int)-1; accs=[]
        for p in ['bruno','caio','paulo']:
            te=(m.pessoa==p).values; tr=~te & (~ext if modo=='sem_ext' else np.ones(len(y),bool))
            w=None
            if modo=='com_ext_peso':
                n0=(y[tr]==0).sum(); n1=(y[tr]==1).sum(); w=np.where(y[tr]==1,n0/n1,1.0)
            mod=XGBClassifier(n_estimators=100,max_depth=2,learning_rate=lr,min_child_weight=1,subsample=.8,colsample_bytree=.8,random_state=42).fit(Xv[tr],y[tr],sample_weight=w)
            P[te]=mod.predict(Xv[te]); accs.append(balanced_accuracy_score(y[te],P[te]))
        i=~ext; print(f"{e:17s} {modo:13s} treino={int((~ext).sum()*2/3+ (ext.sum() if modo!='sem_ext' else 0))} bal_lopo={np.mean(accs):.3f}±{np.std(accs):.3f} acc={accuracy_score(y[i],P[i]):.3f} f1err={f1_score(y[i],P[i]):.3f} recall_err={confusion_matrix(y[i],P[i])[1,1]/y[i].sum():.2f} cm={confusion_matrix(y[i],P[i]).tolist()}")
