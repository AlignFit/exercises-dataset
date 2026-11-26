import sys,numpy as np,pandas as pd,warnings;warnings.filterwarnings('ignore')
from sklearn.metrics import balanced_accuracy_score,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline; from sklearn.preprocessing import StandardScaler,LabelEncoder; from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression; from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from xgboost import XGBClassifier
XB,MB,XR,MR,XP,MP=pd.read_pickle('feats.pkl')
XP=XP.loc[XB.index]; XR=XR.loc[XB.index]; M=MB
y=LabelEncoder().fit_transform(M.tipo_exercicio); g=M.pessoa.values
FS={'brutos_todas':XR,'brutos_principal':XP,'biomec':XB,'biomec+brutos':pd.concat([XB,XP],axis=1)}
imp=lambda: SimpleImputer(strategy='median')
MD={'xgb_atual':lambda:XGBClassifier(n_estimators=300,max_depth=2,learning_rate=.1,min_child_weight=3,subsample=.8,colsample_bytree=.8,random_state=42),
    'lr':lambda:make_pipeline(imp(),StandardScaler(),LogisticRegression(C=.3,class_weight='balanced',max_iter=5000)),
    'et':lambda:make_pipeline(imp(),ExtraTreesClassifier(500,class_weight='balanced',random_state=0)),
    'rf':lambda:make_pipeline(imp(),RandomForestClassifier(500,class_weight='balanced',random_state=0))}
tr,te=train_test_split(np.arange(len(y)),test_size=.2,random_state=42,stratify=y)
rows=[]
for fn,X in FS.items():
    X=X.values
    for mn,mf in MD.items():
        r={'feats':fn,'modelo':mn}
        p=mf().fit(X[tr],y[tr]).predict(X[te]); r['split_acc']=accuracy_score(y[te],p)
        for grp in ['bruno','caio','paulo','externo']:
            t=g==grp; p=mf().fit(X[~t],y[~t]).predict(X[t]); r[grp]=balanced_accuracy_score(y[t],p)
        r['media_equipe']=np.mean([r['bruno'],r['caio'],r['paulo']]); r['media_4grupos']=np.mean([r[k] for k in ['bruno','caio','paulo','externo']])
        rows.append(r)
R=pd.DataFrame(rows); R.to_csv('opt4.csv',index=False); pd.set_option('display.width',250)
print(R.sort_values('media_4grupos',ascending=False).round(3).to_string(index=False))
