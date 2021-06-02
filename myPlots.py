import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

def confusionMat(b_confusion_bin, b_confusion_multi, m_confusion_bin, m_confusion_multi, path):
    labels_bin = ["0","1"]
    labels_multi = ["1","2","3","4","5","6"]

    fig,((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows = 2, ncols = 2)
    fig.tight_layout(pad=3)

    g1 = sns.heatmap(b_confusion_bin, annot=True, cmap='Blues', fmt='g', ax=ax1)
    ax1.set_xticks(np.arange(len(labels_bin)))
    ax1.set_yticks(np.arange(len(labels_bin)))
    ax1.set_xticklabels(labels_bin)
    ax1.set_yticklabels(labels_bin)
    g1.set_xlabel('Predicted Labels')
    g1.set_ylabel('True Classes')
    ax1.set_title("Binary-corrected pred")

    g2 = sns.heatmap(b_confusion_multi, annot=True, cmap='Blues', fmt='g', ax=ax2)
    ax2.set_xticks(np.arange(len(labels_multi)))
    ax2.set_yticks(np.arange(len(labels_multi)))
    ax2.set_xticklabels(labels_multi)
    ax2.set_yticklabels(labels_multi)
    g2.set_xlabel('Predicted Labels')
    g2.set_ylabel('True Classes')
    ax2.set_title("Binary-multiclass pred")

    g3 = sns.heatmap(m_confusion_bin, annot=True, cmap='Blues', fmt='g', ax=ax3)
    ax3.set_xticks(np.arange(len(labels_bin)))
    ax3.set_yticks(np.arange(len(labels_bin)))
    ax3.set_xticklabels(labels_bin)
    ax3.set_yticklabels(labels_bin)
    g3.set_xlabel('Predicted Labels')
    g3.set_ylabel('True Classes')
    ax3.set_title("Multiclass-binary pred")

    g4 = sns.heatmap(m_confusion_multi, annot=True, cmap='Blues', fmt='g', ax=ax4)
    ax4.set_xticks(np.arange(len(labels_multi)))
    ax4.set_yticks(np.arange(len(labels_multi)))
    ax4.set_xticklabels(labels_multi)
    ax4.set_yticklabels(labels_multi)
    g4.set_xlabel('Predicted Labels')
    g4.set_ylabel('True Classes')
    ax4.set_title("Multiclass pred")

    plt.close(fig)
    fig.savefig(path, dpi=600)

def fitModel(predType, hist, zoomFlag, path):

    fittedModel_loss = hist['loss']
    fittedModel_val_loss = hist['val_loss']
    fittedModel_acc = hist['accuracy']
    fittedModel_val_acc = hist['val_accuracy']

    fig,(ax1, ax2) = plt.subplots(nrows = 1, ncols = 2)
    fig.tight_layout(pad=3)

    ax1.plot(fittedModel_loss, label='loss')
    ax1.plot(fittedModel_val_loss, label = 'val_loss')
    ax1.legend(loc='lower left', fontsize = 7, handlelength = 2)
    if zoomFlag == 0:
        ax1.set_ylim(ymin = 0)
    elif zoomFlag == 2:
        ax1.set_ylim(0.15, 0.45)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    if predType == 'binary':
        ax1.set_title("Binary-corrected pred")
    elif predType == 'multiclass':
        ax1.set_title("Multiclass pred")

    ax2.plot(fittedModel_acc, label='accuracy')
    ax2.plot(fittedModel_val_acc, label = 'val_accuracy')
    ax2.legend(loc='upper left', fontsize = 7, handlelength = 2)
    if zoomFlag == 0:
        ax2.set_ylim(ymax = 1)
    elif zoomFlag == 2:
        ax2.set_ylim(0.8, 0.95)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    if predType == 'binary':
        ax2.set_title("Binary-corrected pred")
    elif predType == 'multiclass':
        ax2.set_title("Multiclass pred")
    
    plt.close(fig)
    fig.savefig(path, dpi=600)