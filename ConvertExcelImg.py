import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np

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


Birds = B.to_numpy()
Fire = F.to_numpy()
Chainsaw = C.to_numpy()
Handsaw = Ha.to_numpy()
Helicopter = He.to_numpy()

data = [Birds, Fire, Chainsaw, Handsaw, Helicopter]
dataName = ["Birds", "Fire", "Chainsaw", "Handsaw", "Helicopter"]
lendata = int(len(Birds) / 20 + len(Fire) / 20 + len(Chainsaw) / 20 + len(Handsaw) / 20 + len(Helicopter) / 20)
XE = np.zeros((1, 20, 16))

for i in range(len(data)):
    for j in range(len(data[i]) // 20):
        print (i, j)
        shapex = len(data[i] // 20)
        temp = data[i][j * 20:(j + 1) * 20, :]
        X1 = temp[:, :64 // 4]
        X2 = temp[:, 64 // 4:2 * 64 // 4]
        X3 = temp[:, 2 * 64 // 4:3 * 64 // 4]
        X4 = temp[:, 3 * 64 // 4:]

        XL = [X1, X2, X3, X4]

        for x in range (4) :
            X = XL[x] / np.linalg.norm(XL[x])
            X = X.astype('float64')

            plt.clf()
            im = plt.imshow(X,aspect="auto",cmap='jet',origin='lower')
            plt.savefig("dataImg/{}/{}.jpg".format(dataName[i], j+x), format="jpg")

            # On charge l'image et on la transforme en tableau contenant les couleurs
            image_entrée = Image.open("dataImg/{}/{}.jpg".format (dataName[i], j+x))
            image = np.asarray(image_entrée)

            # Partie à compléter
            image_sortie = image[58:-52,80:-63]

            # On sauvegarde les images pour pouvoir les afficher
            Image.fromarray(image_sortie).save("dataImg/{}/{}.jpg".format (dataName[i], j+x))