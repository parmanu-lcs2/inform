import matplotlib.pyplot as plt

experts = [0, 1, 2]
intrinsic = [0.507, 0.495, 0.579]
kl = [9.22, 2.83, 4.70]

plt.figure(figsize=(6, 4))
plt.scatter(intrinsic[0], kl[0], s=80, label="E0")
plt.text(x=intrinsic[0] + .005 , y=kl[0] - .5, s='Llama-3.2-1B', fontdict={'size': 16})
plt.scatter(intrinsic[1], kl[1], s=80, label="E1")
plt.text(x=intrinsic[1] + .005 , y=kl[1] - .5, s='Qwen2.5-3B', fontdict={'size': 16})
plt.scatter(intrinsic[2], kl[2], s=80, label="E2")
plt.text(x=intrinsic[2] + .005 , y=kl[2] - .5, s='Mistral-7B', fontdict={'size': 16})

plt.xlabel("Intrinsic Importance", fontsize=18)
plt.ylabel("KL (Masking)", fontsize=18)
plt.grid(True, alpha=0.3)
plt.xlim(left=.45, right=.65)
plt.ylim(bottom=1, top=10)
plt.xticks(fontsize=18, rotation=45)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.savefig("cascade_inform_scatter.pdf")
plt.close()

experts = ["LLaMA-3.2-1B", "Qwen2.5-3B", "Mistral-7B"]
stopping = [
    0.5366379310344828,
    0.1788793103448276,
    0.28448275862068967
]
colors = ["#2179b1",  "#fe7d27", "#319f38",] 
plt.figure(figsize=(6, 4))
plt.bar(experts, stopping, color=colors)

plt.ylabel("Stopping Probability", fontsize=18)
plt.xlabel("Expert", fontsize=18)

plt.ylim(0, .6)
plt.yticks(fontsize=18)
plt.xticks(fontsize=14)

plt.tight_layout()
plt.savefig("stopping_distribution.pdf")
plt.close()
