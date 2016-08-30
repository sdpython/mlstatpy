"""ce module permet de transformer une image en niveau de gris et
de calculer le gradient"""
import pygame
import geometrie as GEO
import copy
import math
import Numeric
                
def grey_level (image) :
    """met une image en niveau de gris,
    l'intensite en un point est la racine carree de la somme
    des carres des intensites des trois couleurs standards"""
    size = image.get_size ()
    for x in xrange (0, size [0]) :
        for y in xrange (0, size [1]) :
            i = image.get_at ((x,y))
            a = i [0]**2 + i [1]**2 + i [2]**2
            a /= 3
            a = int (math.sqrt (a))
            image.set_at ( (x,y), (a,a,a) )
                
                
def calcule_gradient (image) :
    """retourne le gradient d'une image sous forme d'une matrice
    de Point, consideres ici comme des vecteurs"""
    size     = X,Y = image.get_size ()
    m        = [  [ GEO.Point (0,0) for i in xrange (0,Y) ] for j in xrange (0,X) ]
    res      = Numeric.array (m)  # pour voir d'autres types, le type array du package Numeric, 
                                  # plus efficace qu'une liste de listes pour les matrices
 
    for x in xrange (0,size [0] - 1) :
        for y in xrange (0, size [1] - 1) :
            ij = image.get_at ( (x,y) ) [0]     # c'est une image en niveau de gris
            Ij = image.get_at ( (x+1,y) ) [0]   # les trois intensites sont egales
            iJ = image.get_at ( (x,y+1) )  [0]  # on ne prend que la premiere
            IJ = image.get_at ( (x+1,y+1) )  [0]
            gx = 0.5 * (IJ - iJ + Ij - ij)
            gy = 0.5 * (IJ - Ij + iJ - ij)
            res [ (x,y) ] = GEO.Point (gx,gy)            
    return res

def image_gradient (image, gradient, more = None, direction = -1) :
    """construit une image a partir de la matrice de gradient
    afin de pouvoir l'afficher grace au module pygame,
    cette fonction place directement le resultat dans image,
    
    si direction > 0, cette fonction affiche egalement le gradient sur 
    l'image tous les 10 pixels si direction vaut 10"""
    size = X,Y = image.get_size ()
    for x in xrange (0, X-1) :
        for y in xrange (0, Y-1) :
            n = gradient [ (x,y) ]
            if more == None : v = n.norme ()
            elif more == "x" : v = n.x/2 + 127
            else : v = n.y/2 + 127
            image.set_at ( (x,y), (v,v,v) )
    if direction > 0 :
        # on dessine des petits gradients dans l'image
        for x in xrange (0, X, direction) :
            for y in xrange (0, Y, direction) :
                n = gradient [ (x,y) ]
                t = n.norme ()
                if t == 0 : continue
                m = copy.copy (n)
                m.scalairek (1.0 / t)
                if t > direction : t = direction
                if t < 2 : t = 2
                m.scalairek (t)
                pygame.draw.line (image, (255,255,0), (x,y), \
                                (x + int (m.x), y + int (m.y)))
    elif direction == -2 :
        # derniere solution, la couleur represente l'orientation
        # en chaque point de l'image
        for x in xrange (0, X) :
            for y in xrange (0, Y) :
                n = gradient [ (x,y) ]
                i = -n.x * 10 + 128
                j = n.y * 10 + 128
                i,j = min (i, 255), min (j, 255)
                i,j = max (i, 0), max (j,0)
                image.set_at ((x,y), (0,j,i))
        
    return image
