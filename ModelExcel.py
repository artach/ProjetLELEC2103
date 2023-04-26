import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
import pickle
import matplotlib.pyplot as plt
import os, sys
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from seaborn import heatmap
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from numpy import newaxis
import random as random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer


def get_accuracy(prediction, target):
    """
    Compute the accuracy between prediction and ground truth.
    """
    return np.sum(prediction == target) / len(prediction)

def data_aug(X, y, fv_size, time_shift=True, add_noise=True, masking=True):
    shapex, shapey, shapez = X.shape
    X_tot = X.copy()
    y_tot = y.copy()

    ###time shift###
    if time_shift:
        X_shifted = np.zeros((shapex, shapey, shapez))
        for j in range(X.shape[0]):
            vec = X[j, :, :]
            shift = int(random.random() * 0.3 * 64)  # 64 pcq avec fv_size = 16 on voit rien
            vec_shifted = np.roll(vec, shift, axis=1)
            X_shifted[j, :, :] = vec_shifted
        X_tot = np.concatenate((X_tot, X_shifted), axis=0)
        y_tot = np.concatenate((y_tot, y), axis=0)
        X_shifted = np.zeros((shapex, shapey, shapez))
        for j in range(X.shape[0]):
            vec = X[j, :, :]
            shift = int(random.random() * 0.3 * 64)  # 64 pcq avec fv_size = 16 on voit rien
            vec_shifted = np.roll(vec, shift, axis=1)
            X_shifted[j, :, :] = vec_shifted
        X_tot = np.concatenate((X_tot, X_shifted), axis=0)
        y_tot = np.concatenate((y_tot, y), axis=0)
        X_shifted = np.zeros((shapex, shapey, shapez))
        for j in range(X.shape[0]):
            vec = X[j, :, :]
            shift = int(random.random() * 0.3 * 64)  # 64 pcq avec fv_size = 16 on voit rien
            vec_shifted = np.roll(vec, shift, axis=1)
            X_shifted[j, :, :] = vec_shifted
        X_tot = np.concatenate((X_tot, X_shifted), axis=0)
        y_tot = np.concatenate((y_tot, y), axis=0)

        ###add noise###
    if add_noise:
        noise = np.random.rand(shapex, shapey, shapez) * 0.00001 * (1 / np.max(X))
        X_tot = np.concatenate((X_tot, X + noise), axis=0)
        y_tot = np.concatenate((y_tot, y), axis=0)
    ###add noise###
    if add_noise:
        noise = np.random.rand(shapex, shapey, shapez) * 0.00001 * (1 / np.max(X))
        X_tot = np.concatenate((X_tot, X + noise), axis=0)
        y_tot = np.concatenate((y_tot, y), axis=0)

    ###masking###
    if masking:
        max_mask_pct = 0.1
        n_freq_masks = 1
        n_time_masks = 1
        X_masked = np.zeros((shapex, shapey, shapez))
        for j in range(X.shape[0]):
            vec = X[j]
            Nmel, n_steps = vec.shape
            mask_value = np.mean(vec)
            aug_spec = np.copy(vec)  # avoids modifying spec
            freq_mask_param = max_mask_pct * Nmel

            for _ in range(n_freq_masks):
                height = int(np.round(random.random() * freq_mask_param))
                pos_f = np.random.randint(Nmel - height)
                aug_spec[pos_f: pos_f + height, :] = mask_value

            time_mask_param = max_mask_pct * n_steps
            for _ in range(n_time_masks):
                width = int(np.round(random.random() * time_mask_param))
                pos_t = np.random.randint(n_steps - width)
                aug_spec[:, pos_t: pos_t + width] = mask_value
            X_masked[j, :, :] = aug_spec
        X_tot = np.concatenate((X_tot, X_masked), axis=0)
        y_tot = np.concatenate((y_tot, y), axis=0)

    ###masking###
    if masking:
        max_mask_pct = 0.1
        n_freq_masks = 1
        n_time_masks = 1
        X_masked = np.zeros((shapex, shapey, shapez))
        for j in range(X.shape[0]):
            vec = X[j]
            Nmel, n_steps = vec.shape
            mask_value = np.mean(vec)
            aug_spec = np.copy(vec)  # avoids modifying spec
            freq_mask_param = max_mask_pct * Nmel

            for _ in range(n_freq_masks):
                height = int(np.round(random.random() * freq_mask_param))
                pos_f = np.random.randint(Nmel - height)
                aug_spec[pos_f: pos_f + height, :] = mask_value

            time_mask_param = max_mask_pct * n_steps
            for _ in range(n_time_masks):
                width = int(np.round(random.random() * time_mask_param))
                pos_t = np.random.randint(n_steps - width)
                aug_spec[:, pos_t: pos_t + width] = mask_value
            X_masked[j, :, :] = aug_spec
        X_tot = np.concatenate((X_tot, X_masked), axis=0)
        y_tot = np.concatenate((y_tot, y), axis=0)

    ###masking###
    if masking:
        max_mask_pct = 0.1
        n_freq_masks = 1
        n_time_masks = 1
        X_masked = np.zeros((shapex, shapey, shapez))
        for j in range(X.shape[0]):
            vec = X[j]
            Nmel, n_steps = vec.shape
            mask_value = np.mean(vec)
            aug_spec = np.copy(vec)  # avoids modifying spec
            freq_mask_param = max_mask_pct * Nmel

            for _ in range(n_freq_masks):
                height = int(np.round(random.random() * freq_mask_param))
                pos_f = np.random.randint(Nmel - height)
                aug_spec[pos_f: pos_f + height, :] = mask_value

            time_mask_param = max_mask_pct * n_steps
            for _ in range(n_time_masks):
                width = int(np.round(random.random() * time_mask_param))
                pos_t = np.random.randint(n_steps - width)
                aug_spec[:, pos_t: pos_t + width] = mask_value
            X_masked[j, :, :] = aug_spec
        X_tot = np.concatenate((X_tot, X_masked), axis=0)
        y_tot = np.concatenate((y_tot, y), axis=0)
    return X_tot, y_tot


B = "dataExcel/norm-64_ST_barbe/Birds64norm.xlsx"
F = "dataExcel/norm-64_ST_barbe/Fire64norm.xlsx"
C = "dataExcel/norm-64_ST_barbe/Chainsaw64norm.xlsx"
Ha = "dataExcel/norm-64_ST_barbe/Handsaw64norm.xlsx"
He = "dataExcel/norm-64_ST_barbe/Helicopter64norm.xlsx"
bruit = "dataExcel/norm-64_ST_barbe/bruit64norm.xlsx"

B = pd.read_excel(B, header=None).drop(index=0, columns=[0, 1, 2])
F = pd.read_excel(F, header=None).drop(index=0, columns=[0, 1, 2])
C = pd.read_excel(C, header=None).drop(index=0, columns=[0, 1, 2])
Ha = pd.read_excel(Ha, header=None).drop(index=0, columns=[0, 1, 2])
He = pd.read_excel(He, header=None).drop(index=0, columns=[0, 1, 2])

B2 = "dataExcel/norm-64_marco/Birds64norm-3.xlsx"
F2 = "dataExcel/norm-64_marco/Fire64norm-3.xlsx"
C2 = "dataExcel/norm-64_marco/Chainsaw64norm-3.xlsx"
Ha2 = "dataExcel/norm-64_marco/Handsaw64norm-3.xlsx"
He2 = "dataExcel/norm-64_marco/Helicopter64norm-3.xlsx"

B2 = pd.read_excel(B2, header=None).drop(index=0, columns=[0, 1, 2])
F2 = pd.read_excel(F2, header=None).drop(index=0, columns=[0, 1, 2])
C2 = pd.read_excel(C2, header=None).drop(index=0, columns=[0, 1, 2])
Ha2 = pd.read_excel(Ha2, header=None).drop(index=0, columns=[0, 1, 2])
He2 = pd.read_excel(He2, header=None).drop(index=0, columns=[0, 1, 2])

B = pd.concat([B, B2], ignore_index=True)
F = pd.concat([F, F2], ignore_index=True)
C = pd.concat([C, C2], ignore_index=True)
Ha = pd.concat([Ha, Ha2], ignore_index=True)
He = pd.concat([He, He2], ignore_index=True)

# labels = ["Birds","Fire","Chainsaw","Handsaw","Helicopter"]
labels = np.array([0, 1, 2, 3, 4])

Birds = B.to_numpy()
Fire = F.to_numpy()
Chainsaw = C.to_numpy()
Handsaw = Ha.to_numpy()
Helicopter = He.to_numpy()
# bruit = bruit.to_numpy()

lendata = int(len(Birds) / 20 + len(Fire) / 20 + len(Chainsaw) / 20 + len(Handsaw) / 20 + len(Helicopter) / 20)
XE = np.zeros((1, 20, 16))
y = []

data = [Birds, Fire, Chainsaw, Handsaw, Helicopter]

nm = Normalizer()

def addMatrix (XE, x) :
    xzeros = np.zeros((1, 20, 16))
    if (XE.any() == xzeros.any()) :
            XE[0, :, :] = x 
    else :
        XE = np.append (XE, [x], axis=0)
    return XE


var = np.array([])
mean = np.array([])   #condition : < 0.01 delete 30 bad melspecs
std = np.array([])    #condition : < 0.03 delete good melspecs
Energy = np.array([])


for i in range(len(data)):
    for j in range(len(data[i]) // 20):
        shapex = len(data[i] // 20)
        temp = data[i][j * 20:(j + 1) * 20, :]
        X1 = temp[:, :64 // 4]
        X2 = temp[:, 64 // 4:2 * 64 // 4]
        X3 = temp[:, 2 * 64 // 4:3 * 64 // 4]
        X4 = temp[:, 3 * 64 // 4:]

        Xlist = [X1, X2, X3, X4]
        for k in range (len(Xlist)) :
            Energy = np.append (Energy, np.sqrt(np.sum(np.power(Xlist[k], 2))))
            Xlist[k] = Xlist[k] / np.linalg.norm(Xlist[k])
            if ((np.mean(Xlist[k]) >= 0) and (np.var(Xlist[k]) < 0) and (np.std(Xlist[k]) > 0)) :
                Xlist[k] = Xlist[k].astype('float64')      
                print ("Energy : {}".format(np.sqrt(np.sum(np.power(Xlist[k], 2)))))
                print ("std : {}".format(np.std(Xlist[k])))
                print ("Variance : {}".format(np.var(Xlist[k])))
                print ("Mean : {}\n".format(np.mean(Xlist[k])))
                plt.clf()
                im = plt.imshow(Xlist[k],aspect="auto",cmap='jet',origin='lower')
                cb = plt.colorbar()
                plt.title (str(i))
                plt.show()
                continue
            else :         
                var = np.append (var, np.var(Xlist[k]))
                mean = np.append (mean, np.mean(Xlist[k]))
                std = np.append (std, np.std(Xlist[k]))
                XE = addMatrix (XE, Xlist[k])
                y.append(labels[i])

ybis = np.array(y)
XE = XE.astype('float64')
print (np.shape (XE))

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
ax1.hist(var, bins=30, density=True)
ax1.set_title ("Variance of the melspec")
ax2.hist(mean, bins=30, density=True)
ax2.set_title ("Mean of the melspec")
ax3.hist(std, bins=30, density=True)
ax3.set_title ("Std of the melspec")
ax4.hist(Energy, bins=30, density=True)
ax4.set_title ("Energy of the melspec")
plt.show()
"""
for i in range (lendata*4) :
    j = i
    print ("Energy : {}".format(np.sqrt(np.sum(np.power(XE[j], 2)))))
    print ("std : {}".format(np.std(XE[j])))
    print ("Variance : {}".format(np.var(XE[j])))
    print ("Mean : {}\n".format(np.mean(XE[j])))
    plt.clf()
    im = plt.imshow(XE[j],aspect="auto",cmap='jet',origin='lower')
    cb = plt.colorbar()
    plt.title (str(i))
    plt.show()
"""
X_train, X_test, y_train, y_test = train_test_split(XE, ybis, test_size=0.3, stratify=y)
X_train_flat = np.zeros((len(X_train), 16 * 20))
X_test_flat = np.zeros((len(X_test), 16 * 20))

#n_estimators=100, max_features=100 are optimize hyperparameter
tree = RandomForestClassifier(n_estimators=100, max_features=100, n_jobs=-1)

for w in range(len(X_test)):
    X_test_flat[w] = X_test[w].flatten()
for w in range(len(X_train)):
    X_train_flat[w, :] = X_train[w].flatten()
    y_train = np.reshape(y_train, (len(y_train), 1))

# Data augmentation
X_train_aug, y_train_aug = data_aug(X_train, y_train, 16)
X_train_aug_flat = np.zeros((len(X_train_aug), 20 * 16))
for w in range(len(X_train_aug)):
    X_train_aug_flat[w] = X_train_aug[w].flatten()
    y_train_aug_flat = np.reshape(y_train_aug, (len(y_train_aug), 1))
    
# Scaling
sc = StandardScaler()
X_train_scale = sc.fit_transform(X_train_aug_flat)
X_test_scale = sc.transform(X_test_flat)

"""
nm = Normalizer()
X_train_scale = nm.fit_transform(X_train_scale)
X_test_scale = nm.transform(X_test_scale)
"""

# PCA
n = 100
pca = PCA(n_components=n, whiten=True)
X_train_reduced = pca.fit_transform(X_train_scale)
X_test_reduced = pca.transform(X_test_scale)

# make the model
tree.fit(X_train_reduced, y_train_aug_flat.flatten())
prediction_tree = tree.predict(X_test_reduced)
accuracy_tree = get_accuracy(prediction_tree, y_test)

# save the model
pickle.dump(tree, open('pickleExcel/modelnorm.pickle', 'wb'))
# save the scaler
pickle.dump(sc, open('pickleExcel/scalernorm.pickle', 'wb'))
#save the normalizer
pickle.dump(nm,open("pickleExcel/nm.pickle",'wb'))
#save PCA
pickle.dump(pca,open("pickleExcel/pcanorm.pickle",'wb'))
print(accuracy_tree)