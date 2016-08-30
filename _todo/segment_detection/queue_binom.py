"""ce module construit les probabilites d'une loi binomiale B (n,p)"""
import math

def tabule_queue_binom (n,p) :
    """Retourne un dictionnaire dont la cle est couple d'entiers (a,b)
    si t est le resultat, alors t [(a,b)] est la probabilite 
    qu'il y ait b evenements parmi a sachant que la probabilite d'un
    evenement est p : t [ (a,b) ] = C_a^b p^b (1-p)^(a-b)
    
    pour aller plus vite, ces probabilites sont estimees par recurrence
        forall m, t [(m,0)]   = 1.0
        forall m, t [(m,m+1)] = 0.0
        et t[(m,k)] = p * t [ (m-1, k-1)] + (1-p) * t [ (m-1,k) ]
        
    cette fonction calcule tous les coefficients t [ (a,b) ] pour une 
    probabilite p donnee et b <= a <= n
    
    ces probabilites sont stockees dans un dictionnaire car s'ils etaient 
    stockees dans une matrice, celle-ci serait triangulaire inferieure"""
    t = {}
    t [(0,0)] = 1.0
    t [(0,1)] = 0.0
    for m in xrange(1,n+1):
        t [(m,0)]   = 1.0
        t [(m,m+1)] = 0.0
        for k in xrange(1,m+1):
            t[(m,k)] = p * t [ (m-1, k-1)] + (1-p) * t [ (m-1,k) ]
    return t

