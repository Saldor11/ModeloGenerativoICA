# -*- coding: utf-8 -*-


import os
import cv2
import matplotlib.pyplot as plt

# Ruta de la carpeta que contiene las imágenes
#folder_path = 'C:/Users/JSALVADORRC/Desktop/T-sis Alterada/EsqueletosPrueba5/Transformadas1'  # Cambia esta ruta por la correcta



def Encuadro(imgstr, nombre, kernelz, pnts_clave, img):
    # Cargar la imagen en escala de grises
    img_gray = cv2.imread(imgstr, cv2.IMREAD_GRAYSCALE)
    img_gray1 = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

 
    # Convertir la imagen en escala de grises a RGB (3 canales)
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    img_rgb1 = cv2.cvtColor(img_gray1, cv2.COLOR_GRAY2RGB)
    kernelz1 = kernelz
    
# Iterar sobre los puntos (centros) de los rectángulos
    for (cx, cy) in pnts_clave:
        # Calcular las coordenadas de las esquinas del rectángulo
        x1 = int(cx - kernelz1 / 2)  # Esquina superior izquierda x
        y1 = int(cy - kernelz1 / 2)   # Esquina superior izquierda y
        x2 = int(cx + kernelz1 / 2)  # Esquina inferior derecha x
        y2 = int(cy + kernelz1 / 2)   # Esquina inferior derecha y

        # Color del rectángulo (Rojo en formato BGR)
        color = (0, 0, 255)  # BGR para OpenCV: (Blue, Green, Red)
        color1 = (255, 0, 0)

        # Dibujar el rectángulo en la imagen RGB
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color,1)
        cv2.rectangle(img_rgb1, (x1, y1), (x2, y2), color1,1)
    
  
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
    # Visualizar el resultado
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(img_rgb1)
    ax1.set_title(f"{nombre}")
    ax1.axis('Off')
    ax2.imshow(img_rgb)
    ax2.set_title(f'Imagen transformada')
    ax2.axis('Off')
    plt.tight_layout()

    # # Mostrar la imagen con el rectángulo
    # plt.imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))  # Convertir BGR a RGB para matplotlib
    # plt.axis('off')  # Ocultar los ejes
    # plt.title(f"{nombre}")
    plt.show()
    return img_rgb, img_rgb1

        

