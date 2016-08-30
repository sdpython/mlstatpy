"""ce module determine si un segment est significatif, c'est a dire
si le nombre de fausses alarmes n'est pas trop eleve"""
import queue_binom as QB
import geometrie as GEO
import copy

# on recupere les parametres
import detection_param as DP
param = DP.Parametre ()

class SegmentNFA (GEO.Segment) :
    """un segment + un nombre de fausses alarmes, 
    servira a memoriser les segments significatifs"""    
    
    # voir la classe Point pour __slots__ 
    __slots__ = "nfa"
    
    def __init__ (self, p1,p2,nfa) :
        """constructeur, segment + nombre de fausses alarmes"""
        GEO.Segment.__init__ (self, p1,p2)
        self.nfa = nfa
        
    def __str__ (self) :
        """permet d'afficher ce segment"""
        s = GEO.Segment.__str__ (self)
        s += "  nfa = " + str (self.nfa)
        return s



class InformationPoint (object) :
    """pour retenir toutes les informations relatives a un segment,
    une position (pos), 
    la norme du gradient (norme),
    une information permettant de savoir si le gradient est 
    proche du vecteur normal au segment (aligne)"""

    
    # voir la classe Point pour __slots__ 
    __slots__ = "pos", "aligne", "norme"
    
    def __init__ (self, pos, aligne, norme) :
        """constructeur, initialisation"""
        self.pos, self.aligne, self.norme = pos, aligne, norme
        
    def __str__ (self) :
        """permet d'afficher cette classe"""
        s = "aligne " + str (self.aligne)
        s += "  pix " + str (self.pos)
        s += "  gnor " + str (self.norme)
        return s

class LigneGradient :
    """stocke toutes les informations relatives a un segment de l'image
    reliant deux points du contour, recoit les informations
    de la methode decoupe_gradient,
    
    a partir de la, un segment significatif a deux extremites 
    dont le gradient est dans le bon sens, on parcourt donc 
    tous les couples d'extremites possibles, 
    d'abord la premiere (methode premier_chemin),
    puis les suivant (methode next_chemin) jusqu'au dernier couple
    """

    def __init__ (self, info_ligne) :
        """constructeur"""
        self.info_ligne = info_ligne        # informations
        self.nb         = len (info_ligne)  # nombre de pixels

    def __len__ (self) :
        """retourne le nombre de pixels dans le segment,
        peut etre different de la liste self.info_ligne,
        self.nb est determine par decoupe_gradient"""
        return self.nb
        
    def extremite (self) :
        """comptabilise les indices des extremites possibles,
        les pixels choisis ont un gradient de la bonne orientation"""
        ext = []
        for i in xrange (0, len (self)) :
            if self.info_ligne [i].aligne and \
                (i == 0 or i == len (self)-1 or not self.info_ligne [i-1].aligne or not self.info_ligne [i+1].aligne) :
                ext.append (i)
        return ext
        
    def premier_chemin (self, ext) :
        """retourne la premiere d'extremite possible"""
        return (0,1)
        
    def next_chemin (self, ext, ij) :
        """retourne le couple suivant d'extremites possibles,
        None, dans le cas contraire"""
        if ij [1]  < len (ext)-1 : return   (ij[0],     ij[1]+1)
        elif ij[0] < len (ext)-2 : return   (ij[0]+1,   ij[0]+2)
        else : return None
        
    def calcule_NFA (self, ext, ij, binomiale, nb_seg) :
        """ext [ij[0]] : premier indice du segment,
        ext [ij[1]] : dernier indice du segment,
        calcule le nombre de NFA de ce segment (nombre de fausses alarmes),
        """
        l = 0
        n = 0
        for i in xrange (ext[ij[0]],ext[ij[1]]+1) :
            if self.info_ligne [i].norme < param.seuil_norme : 
                # on evite les petits gradients
                continue
            l += 1
            if self.info_ligne [i].aligne : 
                # on calcule un gradient dans le bon sens
                n += 1
        
        # on determine ensuite la probabilite d'un tel agencement
        # de fausses alarmes
        nfa  = binomiale [(l,n)]
        nfa *= nb_seg
        return nfa
        
    def segments_significatifs (self, binomiale, nb_seg) :
        """comptabilise le nombre de segments significatifs sur une ligne
        et les memorise"""
        
        # on recense les extremites possibles
        ext = self.extremite ()
        
        if len (ext) < 2 : 
            # s'il n'y a qu'une extremite possible, 
            # ce n'est pas suffisant pour faire un segment
            return []
            
        # premier couple d'extremites
        ij  = self.premier_chemin (ext)
        res = []       # pour memoriser les segments significatifs
        
        while ij != None :  # tant qu'il reste un couple d'extremite
            # probabilite de fausses alarmes pour ce segment       
            nfa = self.calcule_NFA (ext, ij, binomiale, nb_seg)
            
            if nfa < param.seuil_nfa :
                # si cette proba est suffisamment faible,
                # l'agencement est un cas rare (non aleatoire),
                # il est significatif
                seg = SegmentNFA ( self.info_ligne [ ext [ ij[0]] ].pos, \
                                     self.info_ligne [ ext [ ij[1]] ].pos, \
                                    nfa) 
                # on l'ajoute a la liste
                res.append ( seg )
                
            # on passe au segment suivant
            ij = self.next_chemin (ext, ij)
          
        # fin
        return res
        
