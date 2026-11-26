import sys, json, numpy as np, pandas as pd, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, confusion_matrix, classification_report, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
D=sys.argv[1]; out={}
df0=pd.read_csv(D)
FC=[c for c in df0.columns if c.endswith('_x') or c.endswith('_y')]
def agg(fr,key):
    g=fr.groupby(key); return pd.concat([g[FC].mean(),g[FC].std().fillna(0),g[FC].min(),g[FC].max()],axis=1)
# ---------------- EXECUÇÃO ----------------
HP={'agachamento':.1,'biceps':.03,'desenvolvimento':.1,'elevacao_lateral':.03}
mk=lambda lr,seed=42: XGBClassifier(n_estimators=100,max_depth=2,learning_rate=lr,min_child_weight=1,subsample=.8,colsample_bytree=.8,random_state=seed)
ex={}
for e,lr in HP.items():
    d=df0[(df0.tipo_exercicio==e)&(df0.pessoa!='externo')].copy(); d['label']=d.tipo_execucao.map({'correta':0,'errada':1})
    X=agg(d,'video'); meta=d.groupby('video')[['label','pessoa','sessao']].first().loc[X.index]; y=meta.label.values; Xv=X.values
    r={'n':len(y),'corretas':int((y==0).sum()),'erradas':int((y==1).sum())}
    Xtr,Xte,ytr,yte=train_test_split(Xv,y,test_size=.2,random_state=42,stratify=y)
    p=mk(lr).fit(Xtr,ytr).predict(Xte)
    r['split']={'acc':accuracy_score(yte,p),'f1_err':f1_score(yte,p,pos_label=1,zero_division=0),'n_test':len(yte),'cm':confusion_matrix(yte,p,labels=[0,1]).tolist()}
    accs=[]
    for s in range(10):
        a,b,c,dd=train_test_split(Xv,y,test_size=.2,random_state=s,stratify=y); accs.append(accuracy_score(dd,mk(lr).fit(a,c).predict(b)))
    r['split10']=(float(np.mean(accs)),float(np.std(accs)))
    # LOPO por pessoa
    folds=[];P=np.zeros_like(y)
    for pes in sorted(meta.pessoa.unique()):
        te=(meta.pessoa==pes).values; m=mk(lr).fit(Xv[~te],y[~te]); pp=m.predict(Xv[te]); P[te]=pp
        folds.append({'pessoa':pes,'n':int(te.sum()),'acc':accuracy_score(y[te],pp),'bal':balanced_accuracy_score(y[te],pp),'f1_err':f1_score(y[te],pp,pos_label=1,zero_division=0)})
    r['lopo']={'folds':folds,'acc_mean':float(np.mean([f['acc'] for f in folds])),'acc_std':float(np.std([f['acc'] for f in folds])),'bal_mean':float(np.mean([f['bal'] for f in folds])),'bal_std':float(np.std([f['bal'] for f in folds])),
       'pooled_acc':accuracy_score(y,P),'pooled_f1_err':f1_score(y,P,pos_label=1),'cm':confusion_matrix(y,P).tolist(),'prf':[list(map(float,v)) for v in precision_recall_fscore_support(y,P,labels=[0,1])[:3]]}
    # GroupKFold por sessão
    gk=GroupKFold(n_splits=min(5,meta.sessao.nunique())); Ps=np.zeros_like(y); fa=[]
    for tr,te in gk.split(Xv,y,meta.sessao.values):
        pp=mk(lr).fit(Xv[tr],y[tr]).predict(Xv[te]); Ps[te]=pp; fa.append(accuracy_score(y[te],pp))
    r['sessao']={'acc_mean':float(np.mean(fa)),'acc_std':float(np.std(fa)),'pooled_acc':accuracy_score(y,Ps),'f1_err':f1_score(y,Ps,pos_label=1),'n_sessoes':int(meta.sessao.nunique())}
    ex[e]=r
out['execucao']=ex
# ---------------- EXERCÍCIO ----------------
le=LabelEncoder(); df=df0.copy(); df['label']=le.fit_transform(df.tipo_exercicio); df=df.sort_values(['video_id','person_id','frame'])
def troca(c): return c.replace('esquerd','TMP').replace('direit','esquerd').replace('TMP','direit')
def esp(f):
    m=f.copy()
    for c in FC:
        if c.endswith('_x'): m[c]=np.where(f[c]==0,0,1-f[c])
    return m.rename(columns={c:troca(c) for c in FC})[f.columns]
def ruido(f,rng):
    n=f.copy(); s=rng.uniform(.95,1.05); dl=rng.normal(0,.01,2)
    for c in FC:
        v=n[c].values; dd=dl[0] if c.endswith('_x') else dl[1]; n[c]=np.where(v!=0,(v-.5)*s+.5+dd+rng.normal(0,.005,len(v)),0)
    return n
def variantes(f,rng):
    o=f.groupby(['video_id','person_id']).cumcount()
    return pd.concat([agg(f,'video_id'),agg(esp(f),'video_id'),agg(f[o%2==0],'video_id'),agg(f[o%2==1],'video_id'),agg(ruido(f,rng),'video_id'),agg(ruido(f,rng),'video_id')])
M=lambda: XGBClassifier(n_estimators=300,max_depth=2,learning_rate=.1,min_child_weight=3,subsample=.8,colsample_bytree=.8,random_state=42)
lab=df.groupby('video_id').label.first(); pes=df.groupby('video_id').pessoa.first()
def run(tr_ids,te_ids,aug=True):
    rng=np.random.default_rng(42)
    Xte=agg(df[df.video_id.isin(te_ids)],'video_id'); yte=lab.loc[Xte.index].values
    T=variantes(df[df.video_id.isin(tr_ids)],rng) if aug else agg(df[df.video_id.isin(tr_ids)],'video_id')
    import time; t=time.time(); m=M().fit(T.values,lab.loc[T.index].values); tt=time.time()-t
    t=time.time(); p=m.predict(Xte.values); ti=(time.time()-t)*1000
    return yte,p,tt,ti,len(T)
tr,te=train_test_split(lab.index,test_size=.2,random_state=42,stratify=lab.values)
res={}
for aug in (True,False):
    yte,p,tt,ti,nt=run(tr,te,aug)
    res['aug' if aug else 'noaug']={'acc':accuracy_score(yte,p),'f1':f1_score(yte,p,average='weighted'),'bal':balanced_accuracy_score(yte,p),'n_test':len(yte),'n_train_amostras':nt,'treino_s':tt,'infer_ms':ti,
        'report':classification_report(yte,p,target_names=le.classes_,output_dict=True),'cm':confusion_matrix(yte,p).tolist()}
# test only internos in split
mi=(pes.loc[te]!='externo').values; yte,p,*_=run(tr,te,True); res['aug_internos_no_teste']={'acc':accuracy_score(yte[mi],p[mi]),'n':int(mi.sum())}
lo=[]
for g in ['bruno','caio','paulo','externo']:
    tei=pes.index[pes==g]; tri=pes.index[pes!=g]
    for aug in (True,False):
        yte,p,*_=run(tri,tei,aug); lo.append({'grupo':g,'aug':aug,'n':len(yte),'acc':accuracy_score(yte,p),'bal':balanced_accuracy_score(yte,p)})
res['lopo']=lo; res['classes']=list(le.classes_); res['videos_por_classe']=pd.crosstab(pes.values,lab.map(dict(enumerate(le.classes_))).values).to_dict()
out['exercicio']=res
json.dump(out,open(sys.argv[2],'w'),indent=1,default=float)
print('ok')
