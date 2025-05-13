# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 17:30:48 2025

@author: JSALVADORRC
"""

import os
import numpy as np
from PIL import Image
import cv2
from skimage import morphology
from scipy.ndimage import morphology as morph
from scipy import ndimage
import matplotlib.pyplot as plt


Savimage_path = 'C:/Users/JSALVADORRC/Desktop/T-sis Alterada/Data/TestSeg/Comparaciones/Armadoprocesado'
if not os.path.exists(Savimage_path):
    os.makedirs(Savimage_path)
    

def procesarSkel(imges, names):
    # Definir los kernels para las operaciones morfológicas
    kerneld = np.ones((5, 5), np.uint8)
    kernele = np.ones((3, 3), np.uint8)
    
    # Aplicar dilatación
    img_dilation = cv2.dilate(imges, kerneld, iterations=1)
    img_erosionar = cv2.erode(img_dilation, kernele, iterations=1)
    # Guardar el esqueleto como imagen PNG
    file_path_skel = os.path.join(Savimage_path, f"{names}_procesada.png")
    plt.imsave(file_path_skel, img_erosionar, cmap='gray')

    return img_erosionar #img_combinate