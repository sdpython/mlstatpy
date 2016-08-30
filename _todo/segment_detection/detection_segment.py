#!/usr/bin/python
# coding: utf-8
"""programme principal"""
import pygame
import math
import Numeric as N
import detection_segment_image as DSI
import queue_binom as QB
import detection_param as DP
import geometrie as GEO
import psyco
import copy
import time
import sys
import detection_segment_segangle as DSC
import detection_nfa as NFA
from optparse import OptionParser
import os.path

# pour optimisation, un tiers plus rapide
psyco.full ()

# recupere les parametres de l'algorithme
param = DP.Parametre ()

# initialisation du module pygame qui permet d'afficher les images
pygame.init ()

usage = "usage: %prog [-q] [-f FILE] arg"
parser = OptionParser(usage)
parser.add_option("-f", "--file", dest="filename", default="resultat.jpg",
                  help="ecrit les alignements detectes dans FILE", metavar="FILE")
parser.add_option("-q", "--quiet",
                  action="store_false", dest="verbeux", default=True,
                  help="mode sans animation")
(options, args) = parser.parse_args()
if len(args) != 1:
    args = [ "eglise_zoom2.jpg" ]
    if os.path.exists (args [0]) :
        print "parametres par defaut : ", args 
    else :
        parser.error("nombre d'arguments incorrect")




# chargement le l'image
image_name  = args[0]
image       = pygame.image.load (image_name)
image_copy  = pygame.image.load (image_name)
image_copy2 = pygame.image.load (image_name)

# calcule de l'image en niveau de gris
DSI.grey_level (image)

# on recupere la taille de l'image
size      = image.get_size ()
xx,yy     = size [0], size [1]
print "taille de l'image : ", size

# on calcule les tables de la binomiale pour eviter d'avoir a le fait a 
# chaque fois qu'on en a besoin
nbbin = int(math.ceil(math.sqrt(xx*xx + yy*yy)))
binomiale = QB.tabule_queue_binom (nbbin, param.proba_bin)

# on calcule le gradient en chaque point de l'image
grad    = DSI.calcule_gradient (image)

# on prepare des images permettant d'afficher le gradient
imgrad  = DSI.image_gradient (image_copy, grad)
imgrad2 = DSI.image_gradient (image_copy2, grad, direction = -2)

# on sauve l'image du gradient
pygame.image.save (imgrad, "gradient.jpg")

# nb_seg est le nombre total de segment de l'image
# il y a xx * yy pixels possibles dont (xx*yy)^2 couples de pixels (donc de segments)
nb_seg  = xx * xx * yy * yy

# On affiche une fen�tre blanche
screen    = pygame.display.set_mode( (size [0]*2,size[1]*2))
screen.fill ((255,255,255))

# On affiche un message avant de commencer
font = pygame.font.Font(None, 25)
text = font.render("Cliquer n'importe o� dans l'image pour commencer",True,(0,0,0))
screen.blit(text,(0,text.get_height()))

pygame.display.flip ()

# on attend la pression d'un clic de la souris
DP.attendre_clic (screen)

# on affiche toutes les images a l'ecran (image, gradient)
screen.blit (image, (0,0))
screen.blit (image, (xx,yy))
screen.blit (imgrad, (xx,0))
screen.blit (imgrad2, (0,yy))

# couleurs possibles pour afficher des segments
couleur     = [ (255,0,0), (0,255,0), (0,0,255) ] 

# on cree une instance de la classe permettant de parcourir 
# tous les segments de l'image reliant deux points du contour
seg     = DSC.SegmentBord ( GEO.Point (xx,yy) )

# initialisation avant de parcourir l'image
segment = []                # resultat, ensemble des segments significatifs
ti      = time.clock ()     # memorise l'heure de depart
n       = 0                 # pour savoir combien de segments on a deja visite (seg)
cont    = True              # condition d'arret de la boucle

# on cree une classe permettant de recevoir les informations relatives 
# a l'image et au gradient pour un segment reliant deux points 
# du contour de l'image
ligne   = NFA.LigneGradient ( [ NFA.InformationPoint (GEO.Point (0,0), False, 0) for i in xrange (0, xx + yy) ] )

# premier segment
seg.premier ()

# autres variables a decouvrir en cours de route
clast   = 0
nblast  = 0

# tant qu'on a pas fini
while cont :

    # calcule les informations relative a un segment de l'image reliant deux bords
    # position des pixels, norme du gradient, alignement avec le segment
    seg.decoupe_gradient (grad, param.cos_angle, ligne)

    if len (ligne) > 3 :
        # si le segment contient plus de trois pixels
        # alors on peut se demander s'il inclut des sous-segments significatifs
        res = ligne.segments_significatifs (binomiale, nb_seg)

        # on ajoute les resultats a la liste
        segment.extend (res)

        if (nblast < len (segment)) and options.verbeux :        
            # si de nouveaux segments significatifs ont ete trouves,
            # on les fait apparaitre a l'ecran
            vd = seg.directeur ()
            vd.normalise ()
            c = (0, int (vd.x * 127 + 127), int (vd.y * 127 + 127))
            for ri in xrange (nblast, len (segment)) :
                r = segment [ri]
                a = r.a.arrondi ()
                a = (a.x, a.y)
                b = r.b.arrondi ()
                b = (b.x, b.y)
                pygame.draw.line (screen, c, a,b)
            if options.verbeux: pygame.display.flip ()
            nblast = len (segment)
        
    # on passe au segment suivant
    cont = seg.next ()
    n = n+1
    
    # pour verifier que cela avance
    if n % 1000 == 0 : print "n = ", n, " ... ", len (segment), " temps ", "%2.2f" % (time.clock ()-ti), " sec"
                     
    if n % 10 == 0 :
        # tous les 10 segments, on l'affiche a l'ecran pour 
        # suivre le deroulement du programme a l'ecran
        seg.calcul_bord2 ()
        vd = seg.directeur ()
        vd.normalise ()
        
        # affichage en bas a gauche
        c = (0, int (vd.x * 127 + 127), int (vd.y * 127 + 127))
        a = (int (seg.a.x), int (seg.a.y) + yy)
        b = (int (seg.b.x), int (seg.b.y) + yy)
        pygame.draw.line (screen, c, a,b)
        
        # affichage en haut a droite
        c = int (seg.angle / seg.dangle + 0.5)
        a = (int (seg.a.x) + xx, int (seg.a.y))
        b = (int (seg.b.x) + xx, int (seg.b.y))
        pygame.draw.line (screen, couleur [c % len (couleur)], a,b)
        if options.verbeux: pygame.display.flip ()
        
        if clast != c :
            # si on change d'orientation, on efface les segments affiches
            # pour une meilleure lisibilite
            clast = c
            screen.blit (imgrad, (xx,0))
            screen.blit (imgrad2, (0,yy))
            if options.verbeux: pygame.display.flip ()
        
        
# on affiche le temps necessaire a l'algorithme pour trouver
# tous les segments significatifs
print "temps : ", "%2.2f sec" % (time.clock()-ti)
print "temps : ", "%2.2f min" % ((time.clock()-ti)/60)

# on affiche les segments trouves avec la meme couleur
screen.blit (imgrad, (xx,0))
screen.blit (imgrad2, (0,yy))
if options.verbeux: pygame.display.flip ()
n = 0
for r in segment :
    a = r.a.arrondi ()
    a = (a.x, a.y)
    b = r.b.arrondi ()
    b = (b.x, b.y)
    pygame.draw.line (screen, (255,0,0), a,b)
    n += 1
    if n % 5 == 0 :
        if options.verbeux: pygame.display.flip ()

# on sauve le resultat selon deux formats pour etre sur
pygame.image.save (pygame.display.get_surface (), options.filename)
if options.verbeux: pygame.display.flip ()

while 1:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()

