import sys,numpy as np,pandas as pd,warnings,itertools;warnings.filterwarnings('ignore')
sys.path.insert(0, __import__('os').path.dirname(__file__)); from feats_exploracao import *
from sklearn.metrics import balanced_accuracy_score,f1_score
from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler; from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression; from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier; from sklearn.svm import SVC
from xgboost import XGBClassifier
df=pd.read_csv(sys.argv[1])
XB,MB=video_feats(df); XR,MR=raw_feats(df,False); XP,MP=raw_feats(df,True)
pd.to_pickle((XB,MB,XR,MR,XP,MP),'feats.pkl')
imp=lambda: SimpleImputer(strategy='median')
models={
 'xgb':lambda:XGBClassifier(n_estimators=100,max_depth=2,learning_rate=.05,subsample=.8,colsample_bytree=.5,random_state=42),
 'logreg':lambda:make_pipeline(imp(),StandardScaler(),LogisticRegression(C=.1,class_weight='balanced',max_iter=2000)),
 'svm':lambda:make_pipeline(imp(),StandardScaler(),SVC(C=1,class_weight='balanced')),
 'rf':lambda:make_pipeline(imp(),RandomForestClassifier(300,min_samples_leaf=2,class_weight='balanced',random_state=42)),
 'et':lambda:make_pipeline(imp(),ExtraTreesClassifier(500,min_samples_leaf=2,class_weight='balanced',random_state=42)),
}
rows=[]
for fs,(X,M) in {'brutos_todas':(XR,MR),'brutos_principal':(XP,MP),'biomec':(XB,MB)}.items():
  for e in ['agachamento','biceps','desenvolvimento','elevacao_lateral']:
    s=(M.tipo_exercicio==e).values; Xe=X[s].values; Me=M[s]
    y=(Me.tipo_execucao=='errada').astype(int).values; ext=(Me.pessoa=='externo').values
    for (mn,mf),uext in itertools.product(models.items(),[False,True]):
        P=np.full(len(y),-1); fb=[]
        for p in ['bruno','caio','paulo']:
            te=(Me.pessoa==p).values; tr=~te&(np.ones(len(y),bool) if uext else ~ext)
            m=mf()
            if mn=='xgb':
                w=np.where(y[tr]==1,(y[tr]==0).sum()/max((y[tr]==1).sum(),1),1.); m.fit(Xe[tr],y[tr],sample_weight=w)
            else: m.fit(Xe[tr],y[tr])
            P[te]=m.predict(Xe[te]); fb.append(balanced_accuracy_score(y[te],P[te]))
        i=~ext
        rows.append(dict(feats=fs,ex=e,modelo=mn,externos=uext,bal_fold=np.mean(fb),bal_std=np.std(fb),bal=balanced_accuracy_score(y[i],P[i]),f1err=f1_score(y[i],P[i]),rec_err=(P[i][y[i]==1]==1).mean()))
R=pd.DataFrame(rows); R.to_csv('opt_exec.csv',index=False)
pv=R.pivot_table(index=['feats','modelo','externos'],columns='ex',values='bal_fold'); pv['MEDIA']=pv.mean(1)
pd.set_option('display.width',200); print(pv.sort_values('MEDIA',ascending=False).round(3).head(30))
