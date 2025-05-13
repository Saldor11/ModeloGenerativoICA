# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 04:08:18 2025

@author: JSALVADORRC
"""

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.transforms import v2
import torch
import random
from Encontrarpuntoscerca1x import EnconPntsCerca

# Configuración del dispositivo 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)

# Savimage_path2 = 'C:/Users/JSALVADORRC/Desktop/T-sis Alterada/EsqueletosPrueba5/Transformadas1'
# if not os.path.exists(Savimage_path2):
#     os.makedirs(Savimage_path2)

Savimage_path2 = 'C:/Users/JSALVADORRC/Desktop/T-sis Alterada/Data/TestSeg/Comparaciones/ArmadoTransformsParche'
if not os.path.exists(Savimage_path2):
    os.makedirs(Savimage_path2)
    
Savimage_path3 = 'C:/Users/JSALVADORRC/Desktop/T-sis Alterada/Data/TestSeg/Comparaciones/ArmadoTransformsMaskParche'
if not os.path.exists(Savimage_path3):
    os.makedirs(Savimage_path3)
    
# Fijar la semilla para reproducibilidad 
Valseed = 70
torch.manual_seed(Valseed)
random.seed(Valseed)
np.random.seed(Valseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Valseed)

plt.rcParams['savefig.bbox'] = 'tight'

#kernelz = 60

def transforIm(img, msk, pnt, pntC, kernelz, alpha, npnts, nombre, Nalpha):
    img_Mod = img.copy()
    msk_Mod = msk.copy()
    H, W = img.shape
    kernelz2 = kernelz - 10
    imgImage = Image.fromarray(img)
    mskImage = Image.fromarray(msk)
    elastic_transformer = v2.ElasticTransform(alpha=alpha)
    #pnts_clave1 = EnconPntsCerca(pnt, pntC)
    #random.shuffle(pnts_clave1)
    pnts_clave = pnt[:npnts]
    pnts_clave = random.sample(pnt, npnts)
    
    
    imgregn = []
    mskregn = []
    count = 0
    for x1, y1 in pnts_clave:
        x, y = [y1, x1]
        if x - kernelz // 2 < 0 or x + kernelz // 2 >= img.shape[0] or y - kernelz // 2 < 0 or y + kernelz // 2 >= img.shape[1]:
            continue
        kernelx = elastic_transformer(imgImage)
        kernelx = np.array(kernelx)
        regionx = kernelx[x-kernelz//2:x+kernelz//2+1, y-kernelz//2:y+kernelz//2+1]
        region = img_Mod[x - kernelz // 2:x + kernelz // 2 + 1, y - kernelz // 2:y + kernelz // 2 + 1]
        region = Image.fromarray(region)
        kernel = elastic_transformer(region)
        kernel = np.array(kernel)
        kernel = kernel[5:-5, 5:-5]
        #regionx = regionx[5:-5, 5:-5]
        # Guardar la imagen transformada
        save_path = os.path.join(Savimage_path2, f"{nombre}_{Nalpha}_{count}.png")
        plt.imsave(save_path,kernel, cmap='gray')
        #imgregn.append(kernel)
        
        
        #img_Mod[x - kernelz2 // 2:x + kernelz2 // 2 + 1, y - kernelz2 // 2:y + kernelz2 // 2 + 1] = kernel.copy()
        img_Mod[x - kernelz // 2:x + kernelz // 2 + 1, y - kernelz // 2:y + kernelz // 2 + 1] = regionx.copy()
        
        
        kernelxM = elastic_transformer(mskImage)
        kernelxM = np.array(kernelxM)
        regionxM = kernelxM[x-kernelz//2:x+kernelz//2+1, y-kernelz//2:y+kernelz//2+1]
        region1 = msk_Mod[x - kernelz // 2:x + kernelz // 2 + 1, y - kernelz // 2:y + kernelz // 2 + 1]
        region1 = Image.fromarray(region1)
        kernel1 = elastic_transformer(region1)
        kernel1 = np.array(kernel1)
        #mskregn.append(kernel)
        kernel1 = kernel1[5:-5, 5:-5]
        save_path1 = os.path.join(Savimage_path3, f"{nombre}_{Nalpha}_{count}.png")
        plt.imsave(save_path1,kernel1, cmap='gray')
        #msk_Mod[x - kernelz2 // 2:x + kernelz2 // 2 + 1, y - kernelz2 // 2:y + kernelz2 // 2 + 1] = kernel1.copy()
        msk_Mod[x - kernelz // 2:x + kernelz // 2 + 1, y - kernelz // 2:y + kernelz // 2 + 1] = regionxM.copy()
        count+=1

    # Obtener el rango de valores de la imagen original
    img_min, img_max = img.min(), img.max()

    # Ajustar la imagen transformada al mismo rango que la original
    img_Mod2 = img_Mod.copy()
    img_min_mod, img_max_mod = img_Mod2.min(), img_Mod2.max()

    if img_max_mod - img_min_mod > 0:
        # Normalizar la imagen transformada al rango [0, 1]
        img_Mod2 = (img_Mod2 - img_min_mod) / (img_max_mod - img_min_mod)
        
        # Ajustar la imagen transformada al rango de la imagen original
        img_Mod2 = img_Mod2 * (img_max - img_min) + img_min
    
    # Obtener el rango de valores de la máscara original
    msk_min, msk_max = img.min(), img.max()

    # Ajustar la máscara transformada al mismo rango que la original
    msk_Mod3 = msk_Mod.copy()
    msk_min_mod, msk_max_mod = msk_Mod3.min(), msk_Mod3.max()

    if msk_max_mod - msk_min_mod > 0:
        # Normalizar la máscara transformada al rango [0, 1]
        msk_Mod2 = (msk_Mod3 - msk_min_mod) / (msk_max_mod - msk_min_mod)
        
        # Ajustar la máscara transformada al rango de la imagen original
        msk_Mod3 = msk_Mod3 * (msk_max - msk_min) + msk_min
    
   
    
    # for x1, y1 in pnts_clave:
    #      x, y = [y1, x1]
    #      # Asegurar que el kernel esté dentro de los límites de la imagen
    #      if x - kernelz//2 < 0 or x + kernelz//2 >= img.shape[0] or y - kernelz//2 < 0 or y + kernelz//2 >= img.shape[1]:
    #          continue
         
    #      # #region = kernel[x-kernelz//2:x+kernelz//2+1, y-kernelz//2:y+kernelz//2+1]
    #      # region = img_Mod[x-kernelz//2:x+kernelz//2+1, y-kernelz//2:y+kernelz//2+1]
    #      # region = Image.fromarray(region)
    #      # kernel = elastic_transformer(region)
    #      # kernel = np.array(kernel)
    #      # kernel =kernel[5:-5,5:-5]
    #      # img_Mod[x-kernelz2//2:x+kernelz2//2+1, y-kernelz2//2:y+kernelz2//2+1] = kernel.copy()
         
    #      # Asegurar que las coordenadas están dentro de los límites de la imagen
    #      x_start = max(0, x - kernelz // 2)
    #      y_start = max(0, y - kernelz // 2)
    #      x_end = min(H - 1, x + kernelz // 2)
    #      y_end = min(W - 1, y + kernelz // 2)
        
        
    #      # Dibujar un rectángulo rojo alrededor de la región transformada
    #      cv2.rectangle(img_Mod2, (y_start, x_start), (y_end, x_end), (0, 0, 255), 1)
    # Convertir a uint8 para visualización y guardar
    img_ = (img * 255).astype(np.uint8)
    img_m = (img_Mod2 * 255).astype(np.uint8)
    
    msk_ = (msk * 255).astype(np.uint8)
    msk_m = (msk_Mod2 * 255).astype(np.uint8)
    

    # print(f"\n Original - min: {img.min()}, max: {img.max()}")
    # print(f"Transformada - min: {img_Mod2.min()}, max: {img_Mod2.max()}")    

    # Visualizar el resultado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(img_, cmap='gray')
    ax1.set_title('Imagen')
    ax1.axis('Off')
    ax2.imshow(img_Mod2, cmap='gray')
    ax2.set_title(f'Imagen transformada')
    ax2.axis('Off')
    plt.tight_layout()
    
    # Visualizar el resultado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(msk_, cmap='gray')
    ax1.set_title('Máscara')
    ax1.axis('Off')
    ax2.imshow(msk_Mod3, cmap='gray')
    ax2.set_title(f'Máscara transformada')
    ax2.axis('Off')
    plt.tight_layout()

    # # Guardar la imagen transformada
    # save_path = os.path.join(Savimage_path2, f"{nombre}.png")
    # plt.imsave(save_path, img_Mod2, cmap='gray')
    plt.show()
    return pnts_clave, img_, img_m, msk_, msk_m, #imgregn, mskregn