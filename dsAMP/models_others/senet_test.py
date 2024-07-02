import torch
import sys
import numpy as np
from utils import ProteinDataset, h5py_to_tensor
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, f1_score, confusion_matrix, matthews_corrcoef,accuracy_score
from setnet2 import Classifier3
from tqdm import tqdm
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))
model = Classifier3()
model.to(device)
model.load_state_dict(torch.load('/home/wuyou/project/pythonProject5/pythonProject/pythonProject1/others/SP/best_model_fold_10.pth'))
model.eval()


def create_protein_datasets(train_pos, train_neg):
    X_train_positive = h5py_to_tensor(h5_path=train_pos)
    X_train_negative = h5py_to_tensor(h5_path=train_neg)
    # 将正类和负类的嵌入向量合并
    X_train_combined = np.concatenate((X_train_positive, X_train_negative), axis=0)

    # 创建对应的标签列表
    y_train_combined = [1.0] * len(X_train_positive) + [0.0] * len(X_train_negative)
    # 创建数据集对象
    train_dataset = ProteinDataset(X_train_combined, y_train_combined)
    return train_dataset

train_pos = '/home/wuyou/project/pythonProject5/pythonProject/pythonProject1/others/Streptococcus_pneumoniae_test20.h5'
train_neg = '/home/wuyou/project/pythonProject5/pythonProject/pythonProject1/others/neg_cdhit40_test20.h5'

test_dataset = create_protein_datasets(train_pos, train_neg)

test_loader = DataLoader(test_dataset,batch_size=128, shuffle=False)

all_predits = []
all_labels = []



with torch.no_grad():
    test_bar = tqdm(test_loader, file=sys.stdout)
    for test_data in test_bar:
        test_embeddings, test_labels = test_data
        outputs = model(test_embeddings.to(device))
        predict_y = torch.max(outputs, dim=1)[1]

        all_predits.extend(predict_y.cpu().numpy())
        all_labels.extend(test_labels.cpu().numpy())


accuracy = accuracy_score(all_labels,all_predits)
fpr, tpr, thresholds = roc_curve(all_labels,all_predits)
roc_auc = auc(fpr,tpr)

f1 = f1_score(all_labels,all_predits)

tn, fp, fn, tp = confusion_matrix(all_labels,all_predits).ravel()
cm = confusion_matrix(all_labels,all_predits )
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)

mcc = matthews_corrcoef(all_labels,all_predits )

print('Accuracy:', accuracy)
print("F1 Score:" , f1)
print('Sensitivity (Recall):' , sensitivity)
print('Specificity:', specificity)
print('Precision:', precision)
print('MCC:', mcc)
print('AUC:', roc_auc)

plt.figure()
plt.plot(fpr, tpr, color = 'darkorange', lw=2,label=f'ROC curve (area = { roc_auc:.2f}')
plt.plot([0,1],[0,1],color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic(ROC)')
plt.legend(loc = 'lower right')
plt.show
plt.savefig('ROC.png')

plt.figure(figsize = (8, 6))
sns.set(font_scale = 1.4)
sns.heatmap(cm, annot=True,fmt='g',cmap='Blues', cbar=False,
            xticklabels=['Negtive','Positive'], yticklabels=['Negtive','Positive'])
plt.xlabel('Predicted Lables')
plt.ylabel('True Lables')
plt.title('Confusion Matrix')
plt.show
plt.savefig('Confusion Matrix.png')
