import pandas as pd
from Bio import SeqIO 
from autogluon.text import TextPredictor
from sklearn.manifold import TSNE
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

test_dataset = []
for fa in SeqIO.parse("./test.fasta", "fasta"):
    seq = [" ".join(fa.seq.upper())]
    seq.append(fa.name)
    test_dataset.append(seq)

test_data = pd.DataFrame(test_dataset).rename(columns={0:"sequence",1: "label"})

predictor = TextPredictor.load("./model_9")
embeddings = predictor.extract_embedding(test_data)
X_embedded = TSNE(n_components=2, random_state=123).fit_transform(embeddings)

save_path = 't-SNE.png'
fig, ax = plt.subplots()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

for val, color,kla in [("0", '#03BECA','normal sites'), ("1", '#F77672','lactylation sites')]:
    idx = (test_data['label'].to_numpy() == val).nonzero()
    plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1],s=4,alpha=0.6, c=color, label=kla)
plt.legend(loc='upper right',prop={'family':'simsun', 'size': 10},scatterpoints=1)
plt.savefig(save_path,dpi=600) 