import sys,numpy as np,pandas as pd,warnings;warnings.filterwarnings('ignore')
sys.path.insert(0, __import__('os').path.dirname(__file__))
from sklearn.metrics import balanced_accuracy_score,f1_score
from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler; from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression; from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.svm import SVC
XB,MB,*_=pd.read_pickle('feats.pkl')
imp=lambda: SimpleImputer(strategy='median')
# atributos relevantes por exercício (definidos a priori pela biomecânica, Seção 4.2)
REL={'agachamento':['joelho','quadril','tronco','joelhos_larg','tornoz','joel_torn','quadril_alt'],
     'biceps':['cotovelo','tronco','ombro','cotov_alt','cotov_lat','pulso_alt','ombro_desloc','quadril'],
     'desenvolvimento':['cotovelo','ombro','cotov_lat','cotov_larg','pulso_alt','cotov_alt','pulso_lat','tronco'],
     'elevacao_lateral':['ombro','cotovelo','pulso_alt','cotov_alt','pulso_lat','cotov_lat','tronco']}
def lr(): return make_pipeline(imp(),StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=3000))
def et(s): return make_pipeline(imp(),ExtraTreesClassifier(500,min_samples_leaf=2,class_weight='balanced',random_state=s))
def rf(s): return make_pipeline(imp(),RandomForestClassifier(500,min_samples_leaf=2,class_weight='balanced',random_state=s))
def svm(): return make_pipeline(imp(),StandardScaler(),SVC(C=1,class_weight='balanced',probability=True,random_state=0))
def conf_score(Xtr_c,Xte):
    # conformidade: distância média padronizada ao padrão correto
    m=make_pipeline(imp(),StandardScaler()).fit(Xtr_c); Z=m.transform(Xte); return np.sqrt((np.clip(np.abs(Z),0,6)**2).mean(1))
rows=[]
for e in REL:
    s=(MB.tipo_exercicio==e).values; Me=MB[s]; y=(Me.tipo_execucao=='errada').astype(int).values; ext=(Me.pessoa=='externo').values
    for fsel in ['todos','relevantes']:
        cols=XB.columns if fsel=='todos' else [c for c in XB.columns if any(c.startswith(r) for r in REL[e])]
        Xe=XB.loc[s,cols].values
        for seed in range(5):
            cfg={}
            for name in ['lr','lr_ext','et','et_ext','rf_ext','svm','ens_lr+et_ext','ens_lr+et+svm','conf_ext']:
                cfg[name]=np.full(len(y),np.nan)
            for p in ['bruno','caio','paulo']:
                te=(Me.pessoa==p).values; tri=~te&~ext; tra=~te
                pr={}
                pr['lr']=lr().fit(Xe[tri],y[tri]).predict_proba(Xe[te])[:,1]
                pr['lr_ext']=lr().fit(Xe[tra],y[tra]).predict_proba(Xe[te])[:,1]
                pr['et']=et(seed).fit(Xe[tri],y[tri]).predict_proba(Xe[te])[:,1]
                pr['et_ext']=et(seed).fit(Xe[tra],y[tra]).predict_proba(Xe[te])[:,1]
                pr['rf_ext']=rf(seed).fit(Xe[tra],y[tra]).predict_proba(Xe[te])[:,1]
                pr['svm']=svm().fit(Xe[tri],y[tri]).predict_proba(Xe[te])[:,1]
                pr['ens_lr+et_ext']=(pr['lr']+pr['et_ext'])/2
                pr['ens_lr+et+svm']=(pr['lr']+pr['et_ext']+pr['svm'])/3
                # conformidade: treina só em corretas (equipe-treino + externos); limiar = p90 das corretas internas de treino
                cor=tra&(y==0); sc_te=conf_score(Xe[cor],Xe[te])
                ci=tri&(y==0); sc_tr=conf_score(Xe[cor],Xe[ci]); thr=np.percentile(sc_tr,75)
                pr['conf_ext']=(sc_te>thr).astype(float)
                for k,v in pr.items(): cfg[k][te]=v
            i=~ext
            for k,v in cfg.items():
                pv=(v[i]>=.5).astype(int); yy=y[i]; pes=Me.pessoa.values[i]
                fb=[balanced_accuracy_score(yy[pes==p],pv[pes==p]) for p in ['bruno','caio','paulo']]
                rows.append(dict(ex=e,fsel=fsel,cfg=k,seed=seed,bal_fold=np.mean(fb),bal=balanced_accuracy_score(yy,pv),rec_err=pv[yy==1].mean(),rec_cor=1-pv[yy==0].mean(),f1err=f1_score(yy,pv)))
R=pd.DataFrame(rows); R.to_csv('opt2.csv',index=False)
pd.set_option('display.width',250)
g=R.groupby(['fsel','cfg','ex']).bal_fold.mean().unstack(); g['MEDIA']=g.mean(axis=1)
sd=R.groupby(['fsel','cfg','ex']).bal_fold.std().unstack().mean(axis=1); g['std_seeds']=sd
print(g.sort_values('MEDIA',ascending=False).round(3))
