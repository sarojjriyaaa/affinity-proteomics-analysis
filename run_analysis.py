# Project 3 analysis script (training dataset)
import pandas as pd
from scipy import stats
import numpy as np

raw = pd.read_csv("Raw_Data/olink_style_NPX_raw.csv")
mat = raw.pivot(index="Assay", columns="SampleID", values="NPX")
norm = mat.apply(lambda x: x-x.median(), axis=0)

control = [c for c in norm.columns if c.startswith("C")]
disease = [c for c in norm.columns if c.startswith("D")]

out=[]
for assay in norm.index:
    c=norm.loc[assay,control].dropna()
    d=norm.loc[assay,disease].dropna()
    fc=d.mean()-c.mean()
    t,p=stats.ttest_ind(d,c,equal_var=False,nan_policy="omit")
    out.append([assay,fc,p])
results=pd.DataFrame(out,columns=["Assay","log2FC_Disease_vs_Control","p_value"])
# Benjamini-Hochberg FDR
order=np.argsort(results.p_value.values)
ranked=results.p_value.values[order]*len(results)/(np.arange(len(results))+1)
adj=np.minimum.accumulate(ranked[::-1])[::-1]
fdr=np.empty_like(adj); fdr[order]=np.minimum(adj,1)
results["FDR_BH"]=fdr
results["Significant"]=(results.FDR_BH<0.05)&(results.log2FC_Disease_vs_Control.abs()>=0.5)
results.to_csv("Statistics/differential_proteins.csv",index=False)
