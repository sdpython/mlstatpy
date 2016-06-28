# -*- coding: cp1252 -*-
import os.path, copy, matplotlib, pylab, random, math

def TestROC_sort (a,b) :
    if a [0] < b [0] : return -1
    elif a [0] == b [0] : return 0
    else : return 1


class TestROC (object) :
    """classe permettant de tracer des courbes ROC"""
    
    def clean (self,s) :
        """nettoyage d'une chaîne"""
        s = s.upper () 
        s = s.replace ("_", "")
        s = s.replace ("-", "")
        s = s.replace (".", "")
        s = s.replace (",", "")
        s = s.replace (";", "")
        s = s.replace (":", "")
        s = s.replace ("!", "")
        s = s.replace ("/", "")
        s = s.replace ("\\", "")
        s = s.replace ("'", "")
        s = s.replace ("\"", "")
        s = s.replace ("(", "")
        s = s.replace (")", "")
        return s
    
    def __init__ (self, file) :
        """initialsation avec un fichier texte"""
        if not os.path.exists (file) :
            raise Exception ("fichier absent : " + file)
        f = open (file)
        lines = f.readlines ()
        f.close ()
        self._file = file
        
        self._log = []
        for l in lines :
            try :
                s = l.split (" ")
                if len (s) < 4 : continue
                s [1] = self.clean (s [1])
                s [2] = self.clean (s [2])
                res     = s [1] == s [2]
                score   = float (s [3])
                self._log.append ([score, res])
            except :
                continue
        
        # ce tableau doit être trié
        self._log.sort (TestROC_sort)
        
    def __str__ (self) :
        """affiche les premiers éléments"""
        s = "TestROC : " + self._file + " reco rate : " + ( ("%3.2f") % (self.reco_rate () * 100 )) + "%\n"
        for i in xrange (0,min (5,len (self._log))) :
            s += "      " + str (i) + "\t" + str (self._log [i][0]) + "\t" + str (self._log [i][1]) + "\n"
        for i in xrange (max (len(self._log)-5,0), len (self._log)) :
            s += "      " + str (i) + "\t" + str (self._log [i][0]) + "\t" + str (self._log [i][1]) + "\n"
        s += "      ----------------------------------------------\n"
        roc = self.ROC (10, True)
        s += "      read rate\terror rate\n"
        for r in roc :
            s +=  "      " + ("%3.2f" % (r [0] * 100)) + " %\t" + ("%3.2f" % (r [1] * 100)) + " %\n"
        s += "      ----------------------------------------------\n"
        roc = self.ROC (10, False)
        s += "      reco rate\t error rate\n"
        for r in roc :
            s +=  "      " + ("%3.2f" % (r [0] * 100)) + " %\t" + ("%3.2f" % (r [1] * 100)) + " %\n"
        
        return s
        
    def reco_rate (self) :
        """calcule le taux de reconnaissance"""
        nb = 0
        for l in self._log :
            if l [1] : nb += 1
        return float (nb) / len (self._log)
        
    def ROC (self, nb = 100, read = True, bootstrap = False) :
        """calcule une courbe ROC avec nb points seuils, si nb == -1, autant de points de seuil que d'observations,
        si bootstrap == True, tire aléatoire des nombres pour créer une zone d'intervalle de confiance"""
        
        if not bootstrap : cloud = self._log
        else : cloud = self.random_cloud ()
                    
        # sélection des seuils
        nb      = min (nb, len (cloud))
        seuil   = []
        for i in xrange (0,nb) :
            j = len (cloud) * i / nb
            seuil.append (cloud [j][0])
            
        # on trace la courbe
        roc     = []
        s       = len (seuil)-1
        current = [0,0]
        for ind in xrange (len (cloud)-1, -1, -1)  :
            l = cloud [ind]
            if (l [0] < seuil [s]) and s > 0 : 
                roc.append (copy.copy (current))
                s -= 1
            current [0] += 1
            if not l [1] : current [1] += 1
        if current [0] != 0 :
            roc.append (copy.copy (current))
        roc.reverse ()
            
        # stat
        if read :
            for l in roc :
                if l [0] > 0 :
                    l [1] = float (l [1]) / float (l [0])
                    l [0] = float (l [0]) / float (len (cloud))
        else :
            good, wrong = 0,0
            for l in cloud :
                if l [1] : good += 1
                else : wrong += 1
                
            for l in roc :
                l [0] -= l [1]
                if good > 0     : l [0] = float (l [0]) / good
                if wrong > 0    : l [1] = float (l [1]) / wrong
        return roc
        
    def DrawROC (self, nblist = [100], read = True, file = None, bootstrap = 0) :
        """trace plusieurs courbes ROC sur le même dessin, si file != None, le fichier est enregistré au format eps"""
            
        ncolor = [ "red", "blue", "green", "black", "orange" ]
        
        if bootstrap <= 0 :
            pylab.title("ROC")
            pylab.xlabel("error rate")
            if read : pylab.ylabel("read rate")
            else :  pylab.ylabel("reco rate")
            pylab.grid (True)
            n = 0
            for s in nblist :
                roc = self.ROC (s, read)
                x   = [ r [1] for r in roc ]
                y   = [ r [0] for r in roc ]
                c   = ncolor [ n % len (ncolor) ]
                pylab.plot (x, y, linewidth=1.0, color = c)
                n = n + 1
            pylab.show ()        
        else :
            pylab.title("ROC")
            pylab.xlabel("error rate")
            if read : pylab.ylabel("read rate")
            else :  pylab.ylabel("reco rate")
            pylab.grid (True)
            n = 0
            for s in nblist :
                c   = ncolor [ n % len (ncolor) ]
                for l in xrange (0, bootstrap) :
                    roc = self.ROC (s, read, bootstrap = True)
                    x   = [ r [1] for r in roc ]
                    y   = [ r [0] for r in roc ]
                    pylab.plot (x, y, linewidth=0.15, color = c)
                    n = n + 1
            pylab.show ()        
            
    def ROC_point (self, roc, error) :
        """détermine un point de la courbe passant par (error, y), y étant à détermine,
        la réponse est retourné par interpolation linéaire"""
        for i in xrange (0,len (roc)) :
            if roc [i][1] <= error : break
                
        if i == len (roc) :
            return 0
            
        p2 = roc [i]
        if i - 1 > 0 : p1 = [1,1]
        else : p1 = roc [i-1]
            
        rate = (error - p1 [1]) / (p2 [1] - p1 [1]) * (p2 [0] - p1 [0]) + p1 [0]
        return rate
        
    def ROC_point_intervalle (self, error, nb, read = True, bootstrap = 10, alpha = 0.05) :
        """détermine un intervalle de confiance pour un taux de lecture pour un taux d'erreur donné,
        retourne un taux d'erreur et un intervalle de confiance, retourne aussi le couple min,max,
        cette troisème liste contient aussi moyenne, écart-type, médiance"""
        
        rate = []
        for i in xrange (0, bootstrap) :
            roc = self.ROC (nb, read, bootstrap = True)
            r   = self.ROC_point (roc, error)
            rate.append (r)
            
        rate.sort ()
        
        roc = self.ROC (nb, read)
        ra  = self.ROC_point (roc, error)
        
        i1 = int (alpha * len (rate) / 2)
        i2 = int (min (1.0 - alpha/2 * len (rate) + 0.5, len (rate)-1))
        med = rate [len (rate)/2]
        moy = float (sum (rate)) / len (rate)
        var = 0 
        for r in rate : var += r*r
        var = float (var) / len (rate)
        var = var - moy * moy
        return ra, [ rate [i1], rate [i2] ], [rate [0], rate [len (rate)-1], moy,math.sqrt (var), med]
        
    def random_cloud (self) :
        """tire un nuage aléatoirement"""
        cloud = []
        for i in xrange (0, len (self._log)) :
            k = random.randint (0, len (self._log)-1)
            cloud.append (self._log [k])
        cloud.sort (TestROC_sort)
        return cloud
    
    def split_good_wrong (self, cloud) :
        """retourne deux listes, bon et mauvais scores"""
        good = []
        wrong = []
        for c in cloud :
            if c [1] : good.append (c [0])
            else : wrong.append (c [0])
        return good, wrong
        
    def compute_AUC (self, cloud) :
        """calcule l'aire en-dessous de la courbe"""
        good, wrong = self.split_good_wrong (cloud)
        good.sort ()
        wrong.sort ()
        auc = 0.0
        for b in wrong :
            for a in good :
                if a > b : auc += 1.0
                elif a >= b : auc += 0.5
        n = len (wrong) * len (good)
        if n > 0 : auc /= float (n)
        return auc
    
    def ROC_AUC (self, error, nb, bootstrap = 10, alpha = 0.95) :
        """détermine un intervalle de confiance pour l'aire en dessous de la courbe ROC' par la méthode bootstrap
        retourne un taux d'erreur et un intervalle de confiance, retourne aussi le couple min,max"""
        
        rate = []
        for i in xrange (0, bootstrap) :

            if bootstrap <= 0 : cloud = self._log
            else : cloud = self.random_cloud ()
            auc = self.compute_AUC (cloud)
            rate.append (auc)
            
        rate.sort ()
        
        ra = self.compute_AUC (self._log)
        
        i1 = int (alpha * len (rate) / 2)
        i2 = int (min (1.0 - alpha/2 * len (rate) + 0.5, len (rate)-1))
        med = rate [len (rate)/2]
        moy = float (sum (rate)) / len (rate)
        var = 0 
        for r in rate : var += r*r
        var = float (var) / len (rate)
        var = var - moy * moy
        return ra, [ rate [i1], rate [i2] ], [rate [0], rate [len (rate)-1], moy,math.sqrt (var), med]
    
    
    
if __name__ == "__main__" :
    test = TestROC ("output_sia.txt")
    print (test)

    #test.DrawROC ( [1000])
    
    #test.DrawROC ( [10, 100, 1000, 5000] )
    print ("computing rate..............................")
    rate,inte,mmm = test.ROC_point_intervalle (0.1, 100, read = True, bootstrap = 500)
    print ("rate = \t", "%3.2f" % (rate * 100), "%")
    print ("intervalle à 95% = \t", "[%3.2f, %3.2f]" % (inte [0] * 100,inte [1] * 100))
    print ("intervalle min,max = \t", "[%3.2f, %3.2f]" % (mmm [0] * 100,mmm [1] * 100))
    print ("moyenne = %3.2f, écart-type = %3.2f, médiance = %3.2f" % (mmm [2] * 100,mmm [3] * 100, mmm [4] * 100))

    rate,inte,mmm = test.ROC_AUC (0.1, 100, bootstrap = 200)
    print ("AUC= \t", "%3.2f" % (rate))
    print ("intervalle à 95% = \t", "[%3.2f, %3.2f]" % (inte [0],inte [1]))
    print ("intervalle min,max = \t", "[%3.2f, %3.2f]" % (mmm [0],mmm [1]))
    print ("moyenne = %3.2f, écart-type = %3.2f, médiance = %3.2f" % (mmm [2] * 100,mmm [3] * 100, mmm [4] * 100))
    
    test.DrawROC ( [100], read = True, bootstrap = 100 )
    
    
            
            
        