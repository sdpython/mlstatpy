"""definition de petits elements geometriques tels que les points 
et les segments, implemente egalement des operations standard
telles le produit scalaire entre deux vecteurs, ..."""
import math
import copy

class Point (object) :
    """definit un point de l'image ou un vecteur, 
    deux coordonnees x et y qui sont reelles"""

    # cette ligne permet de dire au langage Python que la classe
    # Point ne contiendra que deux attributs x et y, 
    # la creation de tout autre attribut est alors interdite,
    # pour ce faire, il faut que la classe herite de la classe object,
    # les Point sont de petits objets qui seront souvent utilisee,
    # ce procede permet de reduire le temps passe a les manipuler,
    # sans cette ligne, le programme donne exactement les memes resultats
    # mais est plus lent
    __slots__ = "x","y"
    
    def __init__ (self, x,y) :
        """constructeur"""
        self.x = x
        self.y = y
        
    def __str__ (self) :
        """permet d'afficher un point avec l'instruction print"""
        return str (self.x) + "," + str (self.y)

    def normalise (self) :
        """normalise le vecteur, sa norme devient 1"""
        v = self.x * self.x + self.y * self.y
        v = math.sqrt (v)
        if v > 0 :   # evite les erreurs si sa norme est nulle
            self.x /= v
            self.y /= v
            
    def scalairek (self, k) :
        """mulitplication par un scalaire"""
        self.x *= k
        self.y *= k
        
    def norme (self) :
        """retourne la norme"""
        return math.sqrt (self.x * self.x + self.y * self.y)
        
    def scalaire (self, k) :
        """calcule le produit scalaire"""
        return self.x * k.x + self.y * k.y
        
    def __iadd__ (self, ad) :
        """ajoute un vecteur a celui-ci"""
        self.x += ad.x
        self.y += ad.y
        return self
        
    def __add__ (self, ad) :
        """ajoute un vecteur a celui-ci"""
        return Point (self.x + ad.x, self.y + ad.y)
        
    def arrondi (self) :
        """retourne les coordonnees arrondies a l'entier le plus proche"""
        return Point (int (self.x + 0.5), int (self.y + 0.5))
        
    def __sub__ (self, p) :
        """soustraction de deux de vecteurs"""
        return Point (self.x - p.x, self.y - p.y)
        
    def angle (self) :
        """retourne l'angle du vecteur"""
        return math.atan2 (self.y, self.x)
        
    def __eq__ (self, a) :
        """retourne True si les deux points self et a sont egaux,
        False sinon"""
        return self.x == a.x and self.y == a.y

class Segment (object) :
    """definit un segment, soit deux Point"""
    
    # voir le commentaire associees a la ligne contenant __slots__ 
    # dans la classe Point
    __slots__ = "a","b"

    def __init__ (self, a,b) :
        """constructeur, pour eviter des erreurs d'etourderie,
        on cree des copies des extremites a et b,
        comme ce sont des classes, une simple affectation ne suffit pas"""
        self.a,self.b = copy.copy (a), copy.copy (b)

    def __str__ (self) :
        """permet d'afficher le segment avec l'instruction print"""
        return self.a.__str__ () + " - " + self.b.__str__ ()
        
    def directeur (self) :
        """retourne le vecteur directeur du segment, 
        ce vecteur est norme"""
        p = Point (self.b.x - self.a.x, self.b.y - self.a.y)
        p.normalise ()
        return p

    def normal (self) :
        """retourne le vecteur normal du segment,
        ce vecteur est norme"""
        p = Point (self.a.y - self.b.y, self.b.x - self.a.x)
        p.normalise ()
        return p
        
if __name__ == "__main__" :
    # pour tester, n'est execute que si ce fichier
    # est le programme principal"""
    p  = Point (2,2)
    pp = Point (3,5)    
    print p,pp
    pp += p
    pp += p
    print p,pp
    
    
