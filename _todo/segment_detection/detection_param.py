"""ce module contient la liste des parametres utilises 
pour la detection des lignes dans une image"""
import math

"""ensemble des parametres utilises pour la detection des lignes dans une image
    proba_bin   est en fait un secteur angulaire (360 / 16) qui determine la proximite de deux directions
    cos_angle   est le cosinus de l'angle correspondant a ce secteur angulaire
    seuil_nfa   au dela de ce seuil, on considere qu'un segment genere trop de fausses alertes pour etre selectionne
    seuil_norme norme en deca de laquelle un gradient est trop petit pour etre significatif (c'est du bruit)
    angle       lorsqu'on balaye l'image pour detecter les segments, on tourne en rond selon les angles 0, angle, 2*angle, 3*angle, ..."""
class Parametre :
    def __init__ (self) :    
        self.proba_bin      = 1.0/16
        self.cos_angle      = math.cos (self.proba_bin/2 * (math.pi * 2))
        self.seuil_nfa      = 1e-5
        self.seuil_norme    = 2
        self.angle          = math.pi / 24.0



import pygame
def attendre_clic (screen):
    """attend la pression d'un clic de souris 
    avant de continuer l'execution du programme,
    methode pour pygame"""
    color   = 0,0,0
    pygame.display.flip ()
    reste = True
    while reste:
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONUP :
                reste = False
                break
                