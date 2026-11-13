import csv, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
r=list(csv.DictReader(open("/tmp/finrank_vc_mistral.csv")))
f=lambda k:[float(x[k]) for x in r]
p=f("p_corrupt"); conf=f("sc_confidence"); acc=f("sc_accuracy")
no=f("silent_err_noVerify"); B=f("silent_err_Bcheck"); C=f("silent_err_Cprov")
fig,(a1,a2)=plt.subplots(1,2,figsize=(11.5,4.4))
# left: silent-error sweep + confidence/accuracy
a1.plot(p,no,"o-",color="#7f8c8d",label="no verification")
a1.plot(p,B,"^--",color="#e67e22",label="shared-evidence check (current)")
a1.plot(p,C,"D-",color="#27ae60",lw=2.5,label="independent provenance (ours)")
a1.plot(p,conf,":",color="#c0392b",label="self-consistency confidence")
a1.plot(p,acc,"s:",color="#2c3e50",ms=4,label="self-consistency accuracy")
a1.set_title("(a) FinRank value corruption (Mistral-7B, n=60)\nsilent error vs corrupted-evidence fraction")
a1.set_xlabel("fraction of evidence value-corrupted"); a1.set_ylabel("rate"); a1.set_ylim(-0.03,1.05); a1.legend(fontsize=7.5,loc="center left"); a1.grid(alpha=0.3)
# right: detection rate on corrupted set with 95% CI
labels=["shared-check\n(current)","independent\nprovenance (ours)"]
vals=[0.35,0.77]; lo=[0.23,0.65]; hi=[0.47,0.87]
err=[[vals[0]-lo[0],vals[1]-lo[1]],[hi[0]-vals[0],hi[1]-vals[1]]]
a2.bar(labels,vals,color=["#e67e22","#27ae60"],yerr=err,capsize=8,width=0.55)
for i,v in enumerate(vals): a2.text(i,v+0.04,f"{v:.0%}",ha="center",fontweight="bold")
a2.set_title("(b) Corrupted-evidence detection rate\n(paired McNemar p = 6e-08; C>B on 25/60, B>C on 0)")
a2.set_ylabel("detection rate (95% CI)"); a2.set_ylim(0,1.0); a2.grid(alpha=0.3,axis="y")
fig.tight_layout(); fig.savefig("/tmp/finrank_mistral_figure.png",dpi=150); print("saved")
