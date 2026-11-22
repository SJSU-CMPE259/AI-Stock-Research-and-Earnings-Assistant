import csv, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; import numpy as np
def load(p):
    r=list(csv.DictReader(open(p))); f=lambda k:[float(x[k]) for x in r]
    return f("p_corrupt"),f("sc_accuracy"),f("silent_err_Cprov")
pm,accm,cm=load("/tmp/finrank_vc_mistral.csv")
pr,accr,cr=load("/tmp/finrank_vc_deepseek-r1-8b.csv")
fig,(a1,a2)=plt.subplots(1,2,figsize=(11.5,4.4))
a1.plot(pm,accm,"o-",color="#2c3e50",label="Mistral-7B")
a1.plot(pr,accr,"s--",color="#2980b9",label="DeepSeek-R1-8B")
a1.set_title("(a) Reasoning does not resist value corruption\n(both accuracies collapse toward 0)")
a1.set_xlabel("fraction of evidence value-corrupted"); a1.set_ylabel("self-consistency accuracy"); a1.set_ylim(-0.03,0.8); a1.legend(fontsize=9); a1.grid(alpha=0.3)
# grouped detection bars with CI
labels=["Mistral-7B","DeepSeek-R1-8B"]; x=np.arange(2); w=0.35
B=[0.35,0.68]; Blo=[0.23,0.53]; Bhi=[0.47,0.80]
C=[0.77,0.90]; Clo=[0.65,0.80]; Chi=[0.87,0.97]
Berr=[[B[i]-Blo[i] for i in range(2)],[Bhi[i]-B[i] for i in range(2)]]
Cerr=[[C[i]-Clo[i] for i in range(2)],[Chi[i]-C[i] for i in range(2)]]
a2.bar(x-w/2,B,w,color="#e67e22",yerr=Berr,capsize=6,label="shared-check (current)")
a2.bar(x+w/2,C,w,color="#27ae60",yerr=Cerr,capsize=6,label="provenance (ours)")
for i in range(2):
    a2.text(x[i]-w/2,B[i]+0.03,f"{B[i]:.0%}",ha="center",fontsize=8)
    a2.text(x[i]+w/2,C[i]+0.03,f"{C[i]:.0%}",ha="center",fontsize=8,fontweight="bold")
a2.set_xticks(x); a2.set_xticklabels(labels); a2.set_ylim(0,1.05); a2.set_ylabel("corrupted-evidence detection (95% CI)")
a2.set_title("(b) Provenance > shared-check for BOTH models\n(McNemar p=6e-08 Mistral, p=4e-03 R1; C>B one-directional)")
a2.legend(fontsize=8); a2.grid(alpha=0.3,axis="y")
fig.tight_layout(); fig.savefig("/tmp/finrank_overlay.png",dpi=150); print("saved")
