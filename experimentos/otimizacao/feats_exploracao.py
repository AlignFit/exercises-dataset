import numpy as np, pandas as pd
K=['nariz','olho_esquerdo','olho_direito','orelha_esquerda','orelha_direita','ombro_esquerdo','ombro_direito','cotovelo_esquerdo','cotovelo_direito','pulso_esquerdo','pulso_direito','quadril_esquerdo','quadril_direito','joelho_esquerdo','joelho_direito','tornozelo_esquerdo','tornozelo_direito']
FC=[f'{k}_{a}' for k in K for a in 'xy']
META=['tipo_exercicio','tipo_execucao','pessoa','sessao','tipo_erro']

def pessoa_principal(df):
    d=df.copy()
    P=lambda k: d[[f'{k}_x',f'{k}_y']].values
    sh=(P('ombro_esquerdo')+P('ombro_direito'))/2; hp=(P('quadril_esquerdo')+P('quadril_direito'))/2
    d['_tam']=np.linalg.norm(sh-hp,axis=1)
    s=d.groupby(['video','person_id']).agg(n=('frame','count'),t=('_tam','median')).reset_index()
    s['score']=s.n*s.t
    best=s.loc[s.groupby('video').score.idxmax(),['video','person_id']]
    return d.merge(best,on=['video','person_id']).drop(columns='_tam')

def ang(a,b,c):
    v1=a-b; v2=c-b
    cs=(v1*v2).sum(1)/(np.linalg.norm(v1,axis=1)*np.linalg.norm(v2,axis=1)+1e-9)
    return np.degrees(np.arccos(np.clip(cs,-1,1)))

def frame_feats(d):
    P={k:d[[f'{k}_x',f'{k}_y']].values.astype(float) for k in K}
    for k in K: P[k][(P[k]==0).all(1)]=np.nan
    sh=(P['ombro_esquerdo']+P['ombro_direito'])/2; hp=(P['quadril_esquerdo']+P['quadril_direito'])/2
    esc=np.linalg.norm(sh-hp,axis=1); esc=np.where(esc>1e-3,esc,np.nan)
    F={}
    for s,o in (('esq','esquerdo'),('dir','direito')):
        F[f'joelho_{s}']=ang(P[f'quadril_{o}'],P[f'joelho_{o}'],P[f'tornozelo_{o}'])
        F[f'quadril_{s}']=ang(P[f'ombro_{o}'],P[f'quadril_{o}'],P[f'joelho_{o}'])
        F[f'cotovelo_{s}']=ang(P[f'ombro_{o}'],P[f'cotovelo_{o}'],P[f'pulso_{o}'])
        F[f'ombro_{s}']=ang(P[f'quadril_{o}'],P[f'ombro_{o}'],P[f'cotovelo_{o}'])
        F[f'pulso_alt_{s}']=(P[f'ombro_{o}'][:,1]-P[f'pulso_{o}'][:,1])/esc
        F[f'cotov_alt_{s}']=(P[f'ombro_{o}'][:,1]-P[f'cotovelo_{o}'][:,1])/esc
        F[f'cotov_lat_{s}']=np.abs(P[f'cotovelo_{o}'][:,0]-sh[:,0])/esc
        F[f'pulso_lat_{s}']=np.abs(P[f'pulso_{o}'][:,0]-sh[:,0])/esc
    t=sh-hp
    F['tronco_incl']=np.degrees(np.arctan2(np.abs(t[:,0]),np.abs(t[:,1])))
    F['quadril_alt']=(hp[:,1]-np.nanmedian(hp[:,1]))/esc
    lar_q=np.abs(P['quadril_esquerdo'][:,0]-P['quadril_direito'][:,0])
    F['joelhos_larg']=np.abs(P['joelho_esquerdo'][:,0]-P['joelho_direito'][:,0])/(lar_q+1e-6)
    F['tornoz_larg']=np.abs(P['tornozelo_esquerdo'][:,0]-P['tornozelo_direito'][:,0])/(lar_q+1e-6)
    F['joel_torn']=F['joelhos_larg']/(F['tornoz_larg']+1e-6)
    lar_o=np.abs(P['ombro_esquerdo'][:,0]-P['ombro_direito'][:,0])
    F['cotov_larg']=np.abs(P['cotovelo_esquerdo'][:,0]-P['cotovelo_direito'][:,0])/(lar_o+1e-6)
    F['ombro_desloc']=(sh[:,0]-np.nanmedian(sh[:,0]))/esc
    F=pd.DataFrame(F,index=d.index)
    for b in ['joelho','quadril','cotovelo','ombro','pulso_alt','cotov_alt','cotov_lat','pulso_lat']:
        a,c=F[f'{b}_esq'],F[f'{b}_dir']
        F[f'{b}_med']=pd.concat([a,c],axis=1).mean(1); F[f'{b}_dif']=(a-c).abs()
        F[f'{b}_min']=np.fmin(a,c); F[f'{b}_max']=np.fmax(a,c)
        F=F.drop(columns=[f'{b}_esq',f'{b}_dir'])
    return F

def q10(x): return x.quantile(.1)
def q90(x): return x.quantile(.9)

def video_feats(df):
    d=pessoa_principal(df).sort_values(['video','frame'])
    F=frame_feats(d); F['video']=d.video.values
    g=F.groupby('video'); out=[]
    for nm,fn in [('mean','mean'),('std','std'),('min','min'),('max','max'),('p10',q10),('p90',q90)]:
        a=g.agg(fn); a.columns=[f'{c}__{nm}' for c in a.columns]; out.append(a)
    X=pd.concat(out,axis=1)
    for c in F.columns.drop('video'): X[f'{c}__amp']=X[f'{c}__p90']-X[f'{c}__p10']
    meta=d.groupby('video')[META].first().loc[X.index]
    return X.replace([np.inf,-np.inf],np.nan), meta

def raw_feats(df,principal=True):
    d=pessoa_principal(df) if principal else df
    g=d.groupby('video')
    X=pd.concat([g[FC].mean(),g[FC].std().fillna(0),g[FC].min(),g[FC].max()],axis=1)
    meta=d.groupby('video')[META].first().loc[X.index]
    return X,meta
