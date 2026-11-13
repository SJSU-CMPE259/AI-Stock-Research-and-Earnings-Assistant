import csv, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
rows=list(csv.DictReader(open("/tmp/corruption_sweep.csv")))
p=[float(r["p_corrupt"]) for r in rows]
conf=[float(r["sc_confidence"]) for r in rows]; acc=[float(r["sc_accuracy"]) for r in rows]
no=[float(r["silent_err_noVerify"]) for r in rows]; B=[float(r["silent_err_Bcheck"]) for r in rows]
C=[float(r["silent_err_Cprov"]) for r in rows]
fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.2))
a1.plot(p,conf,"o-",color="#c0392b",label="self-consistency confidence")
a1.plot(p,acc,"s-",color="#2c3e50",label="self-consistency accuracy")
a1.fill_between(p,acc,conf,color="#c0392b",alpha=0.12)
a1.set_title("(a) Self-consistency: false confidence under\ncorrelated evidence errors")
a1.set_xlabel("fraction of shared evidence corrupted"); a1.set_ylabel("rate"); a1.set_ylim(-0.03,1.05); a1.legend(fontsize=8); a1.grid(alpha=0.3)
a2.plot(p,no,"o-",color="#7f8c8d",label="no verification")
a2.plot(p,B,"^--",color="#e67e22",label="shared-evidence check (current)")
a2.plot(p,C,"D-",color="#27ae60",linewidth=2.5,label="independent provenance (ours)")
a2.set_title("(b) Silent-error rate vs corruption\n(independent provenance removes correlated errors)")
a2.set_xlabel("fraction of shared evidence corrupted"); a2.set_ylabel("silent error rate"); a2.set_ylim(-0.03,1.05); a2.legend(fontsize=8); a2.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("/tmp/correlated_error_figure.png",dpi=150); print("saved")
