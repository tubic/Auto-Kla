import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio import SeqIO 
from autogluon.text import TextPredictor
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This code is defining a function plot_confusion_matrix that takes in a list of true labels (y_true) and a list of predicted labels (y_pred), and plots a confusion matrix using matplotlib. The confusion matrix is a table that shows the number of correct and incorrect predictions made by a classifier.
    
    The function has several parameters:
        - y_true: A list of true labels
        - y_pred: A list of predicted labels
        - classes: A list of class labels    
        - normalize: If set to True, the confusion matrix will be normalized so that each row sums to 1.
        - title: The title of the plot. If not provided, the title will be 'Confusion matrix, without normalization' if normalization is off, and 'Normalized confusion matrix' if normalization is on.
        - cmap: The colormap to use for the plot. The default is plt.cm.Blues.
    The function first computes the confusion matrix using the confusion_matrix function from sklearn.metrics, and then creates a plot using matplotlib.pyplot. The plot shows the true labels on the y-axis and the predicted labels on the x-axis. The function returns the Axes object for the plot.
    
    The code at the bottom the function using a list of true labels y_test and a list of predicted labels y_pred, and displays the plot.
    Returns:
        - The Axes object for the plot    
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center",
             rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=90, ha="center",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    ax.tick_params(bottom=False, top=False, left=False, right=False)
    return ax

np.set_printoptions(precision=2)

test_dataset = []
for fa in SeqIO.parse("./test.fasta", "fasta"):
    seq = [" ".join(fa.seq.upper())]
    seq.append(fa.name)
    test_dataset.append(seq)
test_data = pd.DataFrame(test_dataset).rename(columns={0:"sequence",1: "label"})

# load model
predictor = TextPredictor.load("./model_9")
# predict
predicted_label = list(map(int,np.array(predictor.predict(test_data))))

true_label = list(map(int, list(test_data['label'])[:]))

class_names = ['Negative','Positive']
# Plot non-normalized confusion matrix
plot_confusion_matrix(true_label, predicted_label, classes=class_names,
                      title='Confusion matrix')
plt.savefig("Confusion matrix of test's fold 9.png")
	