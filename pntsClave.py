# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 21:27:55 2024

@author: JSALVADORRC
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from PIL import Image
from KernelsShitomasi import kernels
import statistics as sta

#Savimage_path = 'C:/Users/JSALVADORRC/Desktop/T-sis Alterada/EsqueletosPrueba5/EroDilaES'
# if not os.path.exists(Savimage_path):
#     os.makedirs(Savimage_path)

Savimage_path = 'C:/Users/JSALVADORRC/Desktop/T-sis Alterada/Data/TestSeg/Comparaciones/ArmadoPuntos'
if not os.path.exists(Savimage_path):
    os.makedirs(Savimage_path)

# Define a custom convolutional layer using these kernels
class CustomConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(CustomConvLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        # Define a convolutional layer without bias
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, padding=kernel_size//2)
        # Initialize the weights with custom kernels
        with torch.no_grad():
            self.conv.weight.copy_(kernels.unsqueeze(1))  # Add a channel dimension
        #self.pooling =nn.MaxPool3d(kernel_size=(21, 1,1), stride=1)

    def forward(self, x):
        # Perform the convolution operation
        x = self.conv(x)
        #x =self.pooling(x)

        return x

# Create an instance of the custom convolutional layer
custom_conv = CustomConvLayer(in_channels=1, out_channels=len(kernels), kernel_size=len(kernels[0]))

def shi_tomasi_corner_detector(image, max_corners, quality_level, min_distance):
    # Convertir a float32 para mayor precisión en los cálculos
    image = np.float32(image)

    # Calcular los gradientes Ix e Iy usando el operador Sobel
    Ix = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # Calcular los productos de los gradientes
    Ix2 = Ix * Ix
    Iy2 = Iy * Iy
    Ixy = Ix * Iy

    # Aplicar un filtro gaussiano para suavizar
    Ix2 = cv2.GaussianBlur(Ix2, (3, 3), 0)
    Iy2 = cv2.GaussianBlur(Iy2, (3, 3), 0)
    Ixy = cv2.GaussianBlur(Ixy, (3, 3), 0)

    # Calcular la respuesta de esquina usando los autovalores mínimos
    rows, cols = image.shape
    R = np.zeros((rows, cols))
    for y in range(rows):
        for x in range(cols):
            if image[y, x] >= 1:  # Considerar solo celdas con valores diferentes de cero
                M = np.array([[Ix2[y, x], Ixy[y, x]], [Ixy[y, x], Iy2[y, x]]])
                eigenvalues = np.linalg.eigvals(M)
                R[y, x] = min(eigenvalues)
            else:
                R[y, x] = 0  # Asegurarse de que celdas con valor 0 no sean detectadas como esquinas


    # Normalizar y umbralizar la respuesta
    R = cv2.normalize(R, None, 0, 1, cv2.NORM_MINMAX)
    threshold = quality_level * R.max()
    corners = np.argwhere(R > threshold)

    # Ordenar por respuesta y mantener solo los mejores
    corners = sorted(corners, key=lambda x: R[x[0], x[1]], reverse=True)
    corners = np.array(corners)

    # Aplicar supresión no máxima
    def non_max_suppression(corners, R, min_distance):
        if len(corners) == 0:
            return []

        # Crear una máscara para marcar las esquinas seleccionadas
        selected_corners = []
        mask = np.zeros_like(R, dtype=np.uint8)
        for y, x in corners:
            if mask[y, x] == 0:
                selected_corners.append((x, y))
                cv2.circle(mask, (x, y), min_distance, 255, -1)

        return selected_corners

    corners = non_max_suppression(corners, R, min_distance)

    return corners

def puntosclave(imgs,names,max_corners,quality_level,min_distance):
    # Aplicar la convulución al tensor
    ImgconKernels = custom_conv(imgs)

    # reducir los canles a uno solo
    y = torch.mean(ImgconKernels, dim=1, keepdim=True)

    #convertir en arreglo el tensor
    Imgprocesadas = y.squeeze().detach().numpy()

    # # Parámetros para el detector de esquinas Shi-Tomasi
    # max_corners = 100      # Número máximo de esquinas a detectar
    # quality_level = 0.11   # Nivel de calidad mínima
    # min_distance = 15     # Distancia mínima entre las esquinas detectadas

    lenz = len(Imgprocesadas)
    pnts =[]
    pntsCent = []
    counts =0

    # Analizar los puntos para cada imagen
    for idx in range(lenz):
        zimage = Imgprocesadas[idx]
        zimage =np.where(zimage > 1, zimage,0)
        detected_points = []
        sumax = 0
        sumay = 0
        count = 0

        # Normalizar la imagen al rango [0, 255]
        zimage_nor = cv2.normalize(zimage, None, 0, 255, cv2.NORM_MINMAX)
        zimage_nor = np.uint8(zimage_nor)  # Convertir a uint8
        Zimagergb = cv2.cvtColor(zimage_nor, cv2.COLOR_GRAY2BGR)
        
        #-----------------------------------Detectar los puntos en la imagen ------------------------
        
        # Detectar las puntos usando el método Shi-Tomasi
        Pcorners = shi_tomasi_corner_detector(zimage_nor, max_corners, quality_level, min_distance)
        
        # Dibujar círculos en las esquinas detectadas
        for x, y in Pcorners:
            sumax += x
            sumay += y
            count += 1
            detected_points.append((x, y))
            cv2.circle(Zimagergb, (x, y), 3, (0, 255, 0), -1)
        
        #-----------------------------------Dispersión de puntos en x, y ------------------------
        #Obtener la media y el centroide
        meanx = sumax / count
        meany = sumay / count
        mediax = int(meanx)
        mediay = int(meany)
        centroid = (mediax, mediay)
        
        # Dibujar el centroide en la imagen
        cv2.circle(Zimagergb, centroid, 3, (255, 0, 0), -1)
        
        # Guardar los centroides y puntos de cada imagen
        pntsCent.append(centroid)
        pnts.append(detected_points)
        
        # Convertir las listas en arreglos
        arrpnts = np.array(detected_points).astype(int)
        arrcentro = np.ones((len(arrpnts), 2)).astype(int)
        arrcentroid = arrcentro * centroid
        
        # Seleccionar los puntos que esten antes y después del centroide
        val1x = arrpnts[np.where(arrpnts[:,0]<arrcentroid[:,0])]
        val1y = arrpnts[np.where(arrpnts[:,1]<arrcentroid[:,1])]
        val2x = arrpnts[np.where(arrpnts[:,0]>=arrcentroid[:,0])]
        val2y = arrpnts[np.where(arrpnts[:,1]>=arrcentroid[:,1])]
       
        #La diferencia entre xi-xˉ(media)
        dif1x = val1x[:,0] - mediax
        dif1y = val1y[:,1] - mediay
        dif2x = val2x[:,0] - mediax
        dif2y = val2y[:,1] - mediay
        
        #Elevar al cuadrado
        diff1x = np.power(dif1x, 2)
        diff1y = np.power(dif1y, 2)
        diff2x = np.power(dif2x, 2)
        diff2y = np.power(dif2y, 2)
        
        #Hacer la sumatoria
        sum1x = sum(diff1x)
        sum1y = sum(diff1y)
        sum2x = sum(diff2x)
        sum2y = sum(diff2y)
        
        # Calcular la varianza
        var1x = int(sum1x / len(val1x))
        var1y = int(sum1y / len(val1y))
        var2x = int(sum2x / len(val2x))
        var2y = int(sum2y / len(val2y))
        
        # Calcular la desviación 
        std1x = int(np.sqrt(var1x))
        std1y = int(np.sqrt(var1y))
        std2x = int(np.sqrt(var2x))
        std2y = int(np.sqrt(var2y))
        
        # Densidad de puntos de cada lado
        denizq = len(val1x)/len(arrpnts)
        dender = len(val2x)/len(arrpnts)
        
        # Dispersión ponderada para cada lado
        dispizq = int(std1x*denizq)
        dispder = int(std2x*dender)
        
        # dispizq1 = np.std(val1x[:,0])
        # dispder1 = np.std(val2x[:,0])
        
        d1 =np.sqrt((std1x**2)+(std2x**2)).astype(int)
        mdgeox = sta.geometric_mean(val1x[:,0])
        
        
        if denizq < dender:
            #-----------------------------------Cuadrante Izquierdo superior ------------------------
            
            #delimitar el cuadrante de los puntos que se quieren analizar
            arrbool =arrpnts < arrcentroid
            mask = arrpnts[np.where((arrpnts[:,0]<arrcentroid[:,0]) & (arrpnts[:,1]<arrcentroid[:,1]))]
            
            if mask.size == 0:
                continue
            else:
                #Crear arreglos con el tamaño de sólo los puntos que se analizaran
                arrcentro2 = np.ones((len(mask),2)).astype(int)
                arrcentroid2 =arrcentro2*centroid
                
                #Obtener las disntancias en el eje x 
                distancia = arrcentroid2-mask
                
                distf =np.sqrt((distancia[:,0]**2)+(distancia[:,1]**2)).astype(int)
                
                #Encontrar el índice de la distancia más grande
                indice_maximo = np.argmax(distf)
                
                #Convertir el arreglo en tupla y dibujar el punto inicial
                tmask =tuple(mask)
                idxarr = np.where((arrpnts[:, 0] == tmask[indice_maximo][0]) & (arrpnts[:, 1] == tmask[indice_maximo][1]))
                idxarr1 = idxarr[0]
                idxarr1 =idxarr1.item()
                #print(f'El indice es : {idxarr1}')
                
                cv2.circle(Zimagergb, tmask[indice_maximo], 3, (0, 0, 255), -1)
                
                
               
                
                
        else:
            #-----------------------------------Cuadrante Derecho superior ------------------------
            #delimitar el cuadrante de los puntos que se quieren analizar
            arrbool =arrpnts < arrcentroid
            mask = arrpnts[np.where((arrpnts[:,0]>arrcentroid[:,0]) & (arrpnts[:,1]<arrcentroid[:,1]))]
            
            if mask.size == 0:
                continue
            else:
                #Crear arreglos con el tamaño de sólo los puntos que se analizaran
                arrcentro2 = np.ones((len(mask),2)).astype(int)
                arrcentroid2 =arrcentro2*centroid
                
                #Obtener las disntancias en el eje x 
                distancia = mask-arrcentroid2
                
                distf =np.sqrt((distancia[:,0]**2)+(distancia[:,1]**2)).astype(int)
                
                #Encontrar el índice de la distancia más grande
                indice_maximo = np.argmax(distf)
                
                #Convertir el arreglo en tupla y dibujar el punto inicial
                tmask =tuple(mask)
                idxarr = np.where((arrpnts[:, 0] == tmask[indice_maximo][0]) & (arrpnts[:, 1] == tmask[indice_maximo][1]))
                idxarr1 = idxarr[0]
                idxarr1 =idxarr1.item()
               # print(f'El indice es : {idxarr1}')
                
                cv2.circle(Zimagergb, tmask[indice_maximo], 3, (0, 0, 255), -1)
        
        cv2.imwrite(os.path.join(Savimage_path, f"{names[idx]}_puntos.png"), Zimagergb)
        # # Mostrar la imagen resultante
        plt.imshow(Zimagergb)
        plt.title(f'{names[idx]}')
        plt.axis('off')
        plt.show()
        counts+=1
    return pnts, pntsCent
      
