# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 18:21:54 2025

@author: JSALVADORRC
"""

import PIL
import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.nn import functional as F
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from UnetMultiTaskModOriginalAttention import MultiTaskUNET
#from UnetMultiTaskModf3 import MultiTaskUNET
from ProcesarSkls1Armado import procesarSkel
from pntsClave import puntosclave
from TransformLocal_Contorno1X import transforIm
from Encuadrar import Encuadro
from filtrarpuntosoutliers import detectar_outliers
torch.cuda.empty_cache()

model = MultiTaskUNET() 
# Configuración del dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
print(device)



# Fijar la semilla para reproducibilidad 
Valseed = 70
torch.manual_seed(Valseed)
random.seed(Valseed)
np.random.seed(Valseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Valseed)

plt.rcParams['savefig.bbox'] = 'tight'


#image_path = Path('C:/Users/JSALVADORRC/Desktop/T-sis Alterada/EsqueletosPrueba5/EjemploTr')
image_path = Path('C:/Users/JSALVADORRC/Desktop/T-sis Alterada/Data/TestSeg/Comparaciones/Testima')
mask_path = Path('C:/Users/JSALVADORRC/Desktop/T-sis Alterada/Data/TestSeg/Comparaciones/Testmas')
#image_path = Path('C:/Users/JSALVADORRC/Desktop/T-sis Alterada/Data/TestSeg/Comparaciones/64')
 #C:\Users\JSALVADORRC\Desktop\T-sis Alterada\Data\TestSeg\Comparaciones\Testima
save_path = 'C:/Users/JSALVADORRC/Desktop/T-sis Alterada/Modelos'

Savimage_path = 'C:/Users/JSALVADORRC/Desktop/T-sis Alterada/Data/TestSeg/Comparaciones/ArmadoPredSegm'
if not os.path.exists(Savimage_path):
    os.makedirs(Savimage_path)
    
Savimage_path1 = 'C:/Users/JSALVADORRC/Desktop/T-sis Alterada/Data/TestSeg/Comparaciones/ArmadoPredSkel'
if not os.path.exists(Savimage_path1):
    os.makedirs(Savimage_path1)

Savimage_path2 = 'C:/Users/JSALVADORRC/Desktop/T-sis Alterada/Data/TestSeg/Comparaciones'
# if not os.path.exists(Savimage_path2):
#     os.makedirs(Savimage_path2)
    
# Savimage_path3 = 'C:/Users/JSALVADORRC/Desktop/T-sis Alterada/Data/TestSeg/Comparaciones'
# # if not os.path.exists(Savimage_path3):
# #     os.makedirs(Savimage_path3)
    
Savimage_path4 = 'C:/Users/JSALVADORRC/Desktop/T-sis Alterada/Data/TestSeg/Comparaciones/ArmadoTransformsImagenCuadro'
if not os.path.exists(Savimage_path4):
    os.makedirs(Savimage_path4)

Savimage_path5 = 'C:/Users/JSALVADORRC/Desktop/T-sis Alterada/Data/TestSeg/Comparaciones/ArmadoTransformsMaskCuadro'
if not os.path.exists(Savimage_path5):
    os.makedirs(Savimage_path5)

Savimage_path6 = 'C:/Users/JSALVADORRC/Desktop/T-sis Alterada/Data/TestSeg/Comparaciones/ArmadoImaOriginal'
if not os.path.exists(Savimage_path6):
    os.makedirs(Savimage_path6)

Savimage_path7 = 'C:/Users/JSALVADORRC/Desktop/T-sis Alterada/Data/TestSeg/Comparaciones/ArmadoMaskOriginal'
if not os.path.exists(Savimage_path7):
    os.makedirs(Savimage_path7)

Savimage_path8 = 'C:/Users/JSALVADORRC/Desktop/T-sis Alterada/Data/TestSeg/Comparaciones/ArmadoImagenOCuadro'
if not os.path.exists(Savimage_path8):
    os.makedirs(Savimage_path8)
    
Savimage_path9 = 'C:/Users/JSALVADORRC/Desktop/T-sis Alterada/Data/TestSeg/Comparaciones/ArmadoMaskOCuadro'
if not os.path.exists(Savimage_path9):
    os.makedirs(Savimage_path9)
    
# Clase personalizada del Dataset
class DatasetIm(Dataset):
    def __init__(self, data_path, mask1_path, transform=None):
        self.images = sorted(os.listdir(data_path))
        self.masks = sorted(os.listdir(mask1_path)) if mask1_path else None
        self.data_path = data_path
        self.mask1_path = mask1_path
        self.transform = transform or T.ToTensor()  # Agregar un transform por defecto si no se proporciona

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Cargar imagen
        img = Image.open(os.path.join(self.data_path, self.images[idx])).convert('L')
        mks = Image.open(os.path.join(self.mask1_path, self.masks[idx])).convert('L')
     
        # Convertir la imagen a tensor
        if self.transform:
            img = self.transform(img)
        
        if self.transform:
            mks = self.transform(mks)
           
        return img, self.images[idx], mks  # Retornar un tensor y el nombre de la imagen

# Configuración del DataLoader
full_dataset = DatasetIm(image_path, mask_path)
lendataset = len(full_dataset)
BATCH_SIZE = lendataset

train_loader = DataLoader(full_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Obtener un batch
#imgs, names = next(iter(train_loader))
#print("Tamaño del batch de imágenes:", imgs.size())  # Debería imprimir el tamaño del tensor de imágenes
#print("Nombres de las imágenes:", names)  # Debería imprimir los nombres de archivo de las imágenes

# Cargar el mejor modelo guardado para evaluar en el conjunto de prueba

#best_model_MultitaskModf3X1
model.load_state_dict(torch.load(os.path.join(save_path, 'best_model_UnetMultiTask_16INIDLossAttention.pth')))
model.eval()  # Modo de evaluación

# Parámetros para los puntos clave
max_corners = 100      # Número máximo de esquinas a detectar
quality_level = 0.09   # Nivel de calidad mínima
min_distance = 11     # Distancia mínima entre las esquinas detectadas
# Parámetros para la transformación local
kernelz = 55 # tamaño de kernel
alpha=125.0 # parametro de transformación 
npnts = 4 # número de puntos para transformar
nalpha = 'alfa125'

# Evaluación en el conjunto de prueba
def evaluate_test_set(model, train_loader):
    with torch.no_grad():
        test_bar = tqdm(train_loader, desc="Evaluating Test Set")
        procedskel_images = []
        image_names = []
        for imges, names, maskes in test_bar:
            imges = imges.to(device)
            y_hat_seg, y_hat_skel = model(imges)
            nombres = [name.removesuffix(".png").removesuffix(".jpg") for name in names] # Convertir a lista de nombres sin sufijo ".png"
            
            y_hat_sigmoid_seg = torch.sigmoid(y_hat_seg).cpu().detach()
            y_hat_sigmoid_skel = torch.sigmoid(y_hat_skel).cpu().detach()
            
            for i in range(imges.size(0)):
                pred_seg = y_hat_sigmoid_seg[i].squeeze().numpy()
                pred_skel = y_hat_sigmoid_skel[i].squeeze().numpy()
                nombre = nombres[i]  # Extraer el nombre de la imagen correspondiente

                # Visualizar predicciones
                plt.figure(figsize=(20, 10))
                plt.subplot(1, 2, 1)
                plt.imshow(pred_seg, cmap='gray', vmin=0, vmax=1)
                plt.title('Máscara Predicha Segmentación Test')
                plt.axis('off')

                plt.subplot(1, 2, 2)
                plt.imshow(pred_skel, cmap='gray', vmin=0, vmax=1)
                plt.title('Máscara Predicha Esqueleto Test')
                plt.axis('off')

                plt.show()

                # Guardar cada predicción en un archivo separado
                file_path_seg = os.path.join(Savimage_path, f"{nombre}_{nalpha}.png")
                file_path_skel = os.path.join(Savimage_path1, f"{nombre}_{nalpha}.png")
                plt.imsave(file_path_seg, pred_seg, cmap='gray')
                plt.imsave(file_path_skel, pred_skel, cmap='gray')
                #Procesar el esqueleto para la detección de puntos
                procedskel = procesarSkel(pred_skel,nombre)
                procedskel_images.append(procedskel)
                image_names.append(nombre)
            
            #Extraer los puntos del esqueleto
            procedskel_tensor = torch.tensor(np.array(procedskel_images)).unsqueeze(1)
            pnts, pntsCent = puntosclave(procedskel_tensor, image_names,max_corners,quality_level, min_distance)
            PntLis = {}
            imgz1 = {}
            imgregs = {}
            mskz1 = {}
            mskregs = {}
            for i in range(imges.size(0)):
                nombre = f"{nombres[i]}"
                realim = imges[i].cpu().detach().squeeze().numpy()
                realmsk = maskes[i].cpu().detach().squeeze().numpy()
                pnt = pnts[i]
                print(pnt)
                print('\n\n\n')
                pnt1 = detectar_outliers(pnt)
                print(pnt1)
                pntC = pntsCent[i] 
                
                PtnClave, imgz, img_Mod2, mskz, img_Mod3 = transforIm(realim,realmsk, pnt1,pntC, kernelz, alpha, npnts, nombre, nalpha)
                #_, _, img_Mod3 = transforIm(realmsk, pnt, kernelz, alpha, npnts, nombre)
                
                # Guardar la imagen transformada
                
                ruta =f'{Savimage_path2}/ArmadoTransforms{nalpha}'
                if not os.path.exists(ruta):
                    os.makedirs(ruta)
                save_path = os.path.join(ruta, f"{nombre}.png")
                plt.imsave(save_path, img_Mod2, cmap='gray')
                # Guardar la imagen original procesada
                save_pathy = os.path.join(Savimage_path6, f"{nombre}.png")
                plt.imsave(save_pathy, imgz, cmap ='gray')
                
                # Guardar la máscara transformada
                ruta1 =f'{Savimage_path2}/ArmadoTransformsMask{nalpha}'
                if not os.path.exists(ruta1):
                    os.makedirs(ruta1)
                save_path_ = os.path.join(ruta1, f"{nombre}.png")
                plt.imsave(save_path_, img_Mod3, cmap='gray')
                
                save_pathw = os.path.join(Savimage_path7, f"{nombre}.png")
                plt.imsave(save_pathw, mskz, cmap ='gray')
                
                PntLis[nombre]=PtnClave
                imgz1[nombre] = imgz
                #imgregs[nombre] = rgimg
                mskz1[nombre] = mskz
                #mskregs[nombre] = rgmsk
                
                print(f'{nombre}:', PntLis[nombre])
            print('\n')
            # Ruta de la carpeta que contiene las imágenes transformadas
            folder_path = f'{ruta}'
            folder_pathImgO = f'{Savimage_path6}'
            # Obtener todas las imágenes en la carpeta
            image_files = [f for f in os.listdir(folder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            # Ordenar las imágenes para asegurarse de que coincidan con los nombres de PntLis
            image_files.sort()  # Ordena alfabéticamente por nombre
            imgorde = {k: imgz1[k] for k in sorted(imgz1)}
            
            
            for i,image_file in enumerate(image_files):
                # Ordenar los nombres para que estén en el mismo orden
                nombres.sort()
                image_path = os.path.join(folder_path, image_file)  # Crear la ruta completa
                imageO_path = os.path.join(folder_pathImgO, image_file)
                nombre = f"{nombres[i]}"
                print(f'Image_{image_file}')
                print(f'{nombre}:', PntLis[nombre])
                print('\n')
                ImgcuadroT, ImgcuadroO = Encuadro(image_path,image_file,kernelz, PntLis[nombre], imageO_path)
                # Guardar la imagen transformada
                save_pathz = os.path.join(Savimage_path4, f"{nombre}_{nalpha}.png")
                plt.imsave(save_pathz, ImgcuadroT)
                
                save_pathzO = os.path.join(Savimage_path8, f"{nombre}_{nalpha}.png")
                plt.imsave(save_pathzO, ImgcuadroO)
                
                # save_pathy = os.path.join(Savimage_path6, f"{nombre}_Ioriginal.png")
                # plt.imsave(save_pathy, imgorde[nombre], cmap ='gray')
            
            # Ruta de la carpeta que contiene las máscaras transformadas
            folder_path1 = f'{ruta1}'
            folder_pathmskO = f'{Savimage_path7}'
            # Obtener todas las imágenes en la carpeta
            masks_files = [f for f in os.listdir(folder_path1) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            # Ordenar las imágenes para asegurarse de que coincidan con los nombres de PntLis
            masks_files.sort()  # Ordena alfabéticamente por nombre
            mskorde = {k: mskz1[k] for k in sorted(mskz1)}
            
            
            for i,mask_file in enumerate(masks_files):
                # Ordenar los nombres para que estén en el mismo orden
                nombres.sort()
                mask_path = os.path.join(folder_path1, mask_file)  # Crear la ruta completa
                mask_pathmskO = os.path.join(folder_pathmskO, mask_file)
                nombre = f"{nombres[i]}"
                print(f'Máscara_{mask_file}')
                print(f'{nombre}:', PntLis[nombre])
                print('\n')
                MskcuadroT, MskcuadroO = Encuadro(mask_path,mask_file,kernelz, PntLis[nombre], mask_pathmskO)
                save_pathz1 = os.path.join(Savimage_path5, f"{nombre}_{nalpha}.png")
                plt.imsave(save_pathz1, MskcuadroT)
                
                save_pathzO1 = os.path.join(Savimage_path9, f"{nombre}_{nalpha}.png")
                plt.imsave(save_pathzO1, MskcuadroO)
                
                # save_pathw = os.path.join(Savimage_path7, f"{nombre}_Moriginal.png")
                # plt.imsave(save_pathw, mskorde[nombre], cmap ='gray')
            
    
                
                
            
    
            
            
                
            
            

evaluate_test_set(model, train_loader)



