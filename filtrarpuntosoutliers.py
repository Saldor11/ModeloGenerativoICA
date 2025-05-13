# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 01:32:52 2025

@author: JSALVADORRC
"""

import numpy as np
from sklearn.cluster import DBSCAN

def detectar_outliers(puntos, eps=12, min_samples=2):
    if len(puntos) == 0:
        return []  # Si no hay puntos, retorna lista vacía
    
    # Convertir la lista de puntos a un array de NumPy
    puntos = np.array(puntos)
    
    # Aplicar DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(puntos)
    print(len(labels))
    
    # Filtrar solo los puntos que no sean outliers (labels != -1)
    puntos_filtrados = puntos[labels == -1]
    
    # Convertir de nuevo el array filtrado a una lista de tuplas
    puntos_filtrados = [tuple(punto) for punto in puntos_filtrados]
    
    return puntos_filtrados

# pnts = [(309, 388), (364, 238), (219, 133), (183, 157), (359, 193), (265, 92), (338, 207), (243, 166), (325, 100), (387, 214), (334, 247), (235, 353), (458, 303), (311, 359), (457, 426), (481, 317), (413, 205), (63, 270), (430, 317), (145, 314), (189, 119), (303, 200), (491, 279), (303, 249), (472, 338), (382, 447), (295, 410), (292, 395), (326, 368), (168, 88), (433, 342), (206, 94), (277, 181), (325, 162), (371, 401), (258, 189), (335, 133), (185, 91), (312, 110), (379, 223), (438, 429), (123, 354), (220, 192), (384, 199), (388, 342), (475, 303), (102, 81), (395, 438), (348, 148), (291, 428), (446, 291), (99, 242), (314, 436), (294, 145), (189, 104), (281, 134), (294, 90), (455, 257), (314, 125), (356, 181), (451, 387), (324, 229), (141, 281), (396, 165), (166, 212), (401, 207), (402, 186), (421, 253), (455, 225), (326, 215), (181, 190), (262, 115), (392, 179), (311, 229), (389, 331), (314, 400), (321, 175), (290, 157), (264, 128), (224, 181), (364, 163), (202, 292), (168, 227), (380, 175), (263, 177), (236, 337), (309, 180), (346, 137), (345, 118), (472, 283), (442, 411), (361, 253), (132, 200), (201, 116), (424, 203), (346, 196)]
# pntosx = detectar_outliers(pnts)