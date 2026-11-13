import csv, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
def load(p):
    r=list(csv.DictReader(open(p)))
    f=lambda k:[float(x[k]) for x in r]
    return f("p_corrupt"),f("sc_confidence"),f("sc_accuracy"),f("silent_err_noVerify"),f("silent_err_Cprov")
pm,confm,accm,nom,cm=load("/tmp/corruption_sweep_mistral.csv")
pr,confr,accr,nor,cr=load("/tmp/corruption_sweep_deepseek-r1-8b.csv")
fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.3))
# panel a: reasoning doesn't help — both models confidently wrong
a1.plot(pm,confm,"--",color="#c0392b",label="confidence (both models)")
a1.plot(pm,accm,"o-",color="#2c3e50",label="Mistral-7B accuracy")
a1.plot(pr,accr,"s--",color="#2980b9",label="DeepSeek-R1-8B accuracy")
a1.fill_between(pm,accm,confm,color="#c0392b",alpha=0.10)
a1.set_title("(a) Reasoning does not resist correlated errors\n(confidence stays 1.0; both accuracies collapse)")
a1.set_xlabel("fraction of shared evidence corrupted"); a1.set_ylabel("rate"); a1.set_ylim(-0.03,1.05); a1.legend(fontsize=8); a1.grid(alpha=0.3)
# panel b: provenance fixes both
a2.plot(pm,nom,"o-",color="#7f8c8d",label="no verification (both)")
a2.plot(pm,cm,"D-",color="#27ae60",linewidth=2.5,label="independent provenance (both)")
a2.set_title("(b) Independent provenance removes the\ncorrelated errors for BOTH models")
a2.set_xlabel("fraction of shared evidence corrupted"); a2.set_ylabel("silent error rate"); a2.set_ylim(-0.03,1.05); a2.legend(fontsize=8); a2.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("/tmp/reasoning_vs_provenance.png",dpi=150); print("saved")
