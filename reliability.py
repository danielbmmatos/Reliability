# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 15:01:28 2022

@author: Eng. Msc Daniel Barbosa Mapurunga Matos
Phd candidate from Universidade Federal do Rio Grande do sul

"""
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import pandas as pd

#------------------------------------------------------------------------------
#1. Iniciating class
#------------------------------------------------------------------------------
""" 
This Python code is developed to create a simples class to solve reliability
problems with FORM and Monte Carlo Analysis. The codes you will find here are
adaptations of Prof. Phd. Herbert Martins Gomes MATLAB codes.

"""              
class reliability():

#------------------------------------------------------------------------------
#1. Equivalent mean and standard deviation of a given probability distribution
#------------------------------------------------------------------------------ 
 
    def equiv(x,S,mean,dist):
        
          """
          Returns the equivalent mean and std of a distribution on normal space
          
          :type x: ndarray
          :param x: analysis point.
          
          :type mean: ndarray
          :param mean: mean array
          
          :type S: ndarray
          :param S: standard deviation array
          
          :type dist: ndarray
          :param dist: distribution type array
          
          """   
          n = len(x)
          Seq = np.zeros(S.shape)
          meq = np.zeros(mean.shape)
          
          for i in range(n):
              
              if dist[i] == 'normal':
                  
                  Seq[i,i] = S[i,i]
                  meq[i]   = mean[i]
                  
              elif dist[i] == 'lognormal':
              
                  zeta = np.sqrt(np.log(1+(S[i,i]/mean[i])**2))
              
                  lamb = np.log(mean[i])-zeta**2/2
              
                  Seq[i,i] = st.norm.pdf(st.norm.ppf(st.lognorm.cdf(
                      x[i],s= zeta,scale = np.exp(lamb))))/st.lognorm.pdf(
                          x[i],s = zeta, scale = np.exp(lamb))
                  meq[i]   = x[i] - Seq[i,i]*st.norm.ppf(
                      st.lognorm.cdf(x[i],s = zeta,scale = np.exp(lamb)))
                  
              elif dist[i] == 'gumbel': 
                  
                  alpha=np.pi/(S[i,i]*np.sqrt(6))  ;  u = mean[i]-0.5772/alpha;
                  Seq[i,i]=st.norm.pdf(st.norm.ppf(np.exp(
                      -np.exp(-alpha*(x[i]-u)))))/(alpha*np.exp(-alpha*(
                          x[i]-u))*np.exp(-np.exp(-alpha*(x[i]-u))));
                  meq[i]=x[i]-Seq[i,i]*st.norm.ppf(np.exp(-np.exp(-alpha*(
                      x[i]-u))));
                  
                  
          return Seq,meq
#------------------------------------------------------------------------------
#2. Nataf's transformation
#------------------------------------------------------------------------------ 
    def nataf(dist,rho,mean,std):
        
        """
        Returns correlation array on natural space using Nataf's 
        transformation.
        
        See LIU; DER KIUREGHIAN, 1986
        
        :type mean: ndarray
        :param mean: mean array
        
        :type std: ndarray
        :param std: standard deviation array
        
        :type dist: ndarray
        :param dist: distribution type array
        
        :type rho: ndarray
        :param rhot: correlation array
        
        """
        
        cv = std/mean
        
        Rz = np.zeros(rho.shape)
        
        for i in range(len(dist)):
            for j in range(len(dist)):
                
                if dist[i]=='normal' and dist[j] == 'normal':
                        Rz[i,j] = rho[i,j]
                        
                elif dist[i]=='normal' and dist[j] == 'lognormal':
                    
                        Rz[i,j] = rho[i,j]*cv[j]/(np.sqrt(np.log(1+cv[j]**2)))
                        
                elif dist[i]=='lognormal' and dist[j] == 'normal':
                    
                        Rz[i,j] = rho[i,j]*cv[i]/(np.sqrt(np.log(1+cv[i]**2)))
                        
                elif dist[i]=='lognormal' and dist[j] == 'lognormal':
                    if rho[i,j] == 0:
                        Rz[i,j] = 0
                    else:
                        Rz[i,j] = rho[i,j] *np.log(1+rho[i,j]*cv[i]*cv[j])/(
                            rho[i,j]*np.sqrt(np.log(1+cv[i]**2)*np.log(
                                1+cv[j]**2)))
                        
                elif dist[i]=='normal' and dist[j] == 'gumbel':
                    
                        Rz[i,j] = 1.031*rho[i,j]
                        
                elif dist[i]=='gumbel' and dist[j] == 'normal':
                    
                        Rz[i,j] = 1.031*rho[i,j]
                        
                elif dist[i]=='lognormal' and dist[j] == 'gumbel':
                    
                        Rz[i,j] =  rho[i,j]*(1.029 + 0.001 *rho[i,j] + 
                                             0.014*cv[j]+0.004*rho[i,j]**2+
                                             0.2338*cv[j]**2-
                                             0.197*rho[i,j]*cv[j])
                        
                elif dist[i]=='gumbel' and dist[j] == 'lognormal':
                    
                        Rz[i,j] =  rho[i,j]*(1.029 + 0.001 *rho[i,j] + 
                                             0.014*cv[i]+0.004*rho[i,j]**2+
                                             0.2338*cv[i]**2
                                             -0.197*rho[i,j]*cv[i])
                        
                elif dist[i]=='gumbel' and dist[j] == 'gumbel':
                    
                        Rz[i,j] = rho[i,j]*(1.064-0.069*rho[i,j]+
                                            0.005*rho[i,j]**2)
        return Rz
  
#------------------------------------------------------------------------------
#3. First Order Reliability method (FORM)
#------------------------------------------------------------------------------    
    def FORM(FEL,mean,std,dist,rho):
        
        """
        Compute FORM analysis with a given limit state function and its 
        random variables characteristics.
        
        :type FEL: function
        :param FEL: limit state function
        
        :type mean: ndarray
        :param mean: mean array
        
        :type std: ndarray
        :param std: standard deviation array
        
        :type dist: ndarray
        :param dist: distribution type array
        
        :type rho: ndarray
        :param rhot: correlation array
        
        
        """
        n = len(mean)   
        
        S = np.diag(std)
        
        Seq,meq = reliability.equiv(mean,S,mean,dist)
        Rz      = reliability.nataf(dist,rho,mean,std)
        
        
        def x2z(x,meq,Seq,Rz):
            c =np.linalg.cholesky(Seq@Rz@Seq)
            
            z =np.linalg.inv(c)@(x-meq)
            return z
        
        def z2x(z,meq,Seq,Rz):
            c =np.linalg.cholesky(Seq@Rz@Seq)
            x = c@z.T + meq
            
            return x
        
        dz = 1e-4;          #for finite differences method
        zn = np.zeros(n);   #inicial value for x
        crit = 1;           #valor do critério de parada
        tol=1e-3;           #tolerância
        k=0;                #iteration counter
        nfunc=0;            #LSF counter
        
        
        while crit >tol:    #Main loop
            
            k       = k+1
            z       = zn
            x       = z2x(z,meq,Seq,Rz)
            Seq,meq = reliability.equiv(x,S,mean,dist)
            Rz      = reliability.nataf(dist,rho,mean,std)
            zo      = x2z(x,meq,Seq,Rz) 
           
            grad = np.zeros(n);       
            g0   = FEL(z2x(zo,meq,Seq,Rz))
           
            
            nfunc=nfunc+1;       
        
            for i in range(n):                 
                zod    =zo.copy()
                zod[i] =zo[i]+dz;
                
                grad[i]=(FEL(z2x(zod,meq,Seq,Rz))-g0)/dz;
                
                nfunc=nfunc+1;
            #grad[0] = x[1]*Seq[0,0]
            #grad[1] = x[0]*Seq[1,1]
            
            alfa = grad/np.linalg.norm(grad);   
            zn   = (grad/(grad.T@grad))*(grad.T@zo-g0);
            beta = np.sqrt(zn.T@zn);                      
            crit = np.linalg.norm(zn-z); 
            
            print('-------------------------','\n',
                  '           FORM          ','\n',
                  '-------------------------','\n',
                  'iteração:{0:5.0f}'.format(k),'\n',
                  'beta = {0:5.3f}'.format(beta),'\n',
                  'chamadas da FEL = {0:5.3f}'.format(nfunc),'\n')  
        
#------------------------------------------------------------------------------
#4. Transforms array U to X
#------------------------------------------------------------------------------    
  
    def Uc2X(Uc,dist,mw,sw):  
        
        
        """
        Transforms array U to X.
        
        :type Uc: ndarray
        :param Uc: project point array
        
        :type sw: ndarray
        :param sw: standard deviation array
        
        :type mw: ndarray
        :param mw: mean array 
        
        :type dist: ndarray
        :param dist: distribution type array
        
        """     
        Xw = np.zeros(Uc.shape)
        for i in range(len(mw)):
            if dist[i]=='normal':
                Xw[i,:]= st.norm.ppf(Uc[i,:],mw[i],sw[i]);
            elif dist[i] =='lognormal':
                zetaw = np.sqrt(np.log(1+(sw[i]/mw[i])**2));
                lambdaw = np.log(mw[i])-zetaw**2/2;
                Xw[i,:]= st.lognorm.ppf(Uc[i,:],zetaw,scale = np.exp(lambdaw))
            elif dist[i] =='gumbel':
                alfaw = np.pi/(sw[i]*np.sqrt(6));uw = mw[i]-0.5772/alfaw;
                Xw[i,:]= (-np.log(-np.log(Uc[i,:]))/alfaw)+uw;  
        return Xw
#------------------------------------------------------------------------------
#5. Monte Carlo Analysis
#------------------------------------------------------------------------------    
  
        
    def MONTECARLO(FEL,mean,std,dist,rho,n,nl):
        """
        Compute Monte Carlo analysis with a given limit state function and its 
        random variables characteristics.
        
        :type FEL: function
        :param FEL: limit state function
        
        :type mean: ndarray
        :param mean: mean array
        
        :type std: ndarray
        :param std: standard deviation array
        
        :type dist: ndarray
        :param dist: distribution type array
        
        :type rho: ndarray
        :param rhot: correlation array
        
        :type n: int
        :param n: number of simulations within a lot
        
        :type nl: int
        :param nl: number of lots
        
        """      
        
        np.random.seed(1)
        nva=len(mean)

        tol=5E-2     #Tolerancia para o Coef de Variação de Pf

        S = np.diag(std)
        C = S@rho@S


        Rz=reliability.nataf(dist,rho,mean,std)

        L = np.linalg.cholesky(Rz)
        n=2*round(n/2);        
        flag=0;nfunc=0;nfail=0 
        mPf=0.0;CVPf=1.0;
        beta = np.zeros(nl)
        Pf = np.zeros(nl)
        CVPf = np.zeros(nl)
        
        for k in range(1,nl+1):
                U1 =np.random.rand(nva,int(n/2)); 
                
                U  =np.hstack((U1,1-U1));
                Z  =st.norm.ppf(U);     ##################
                Zc = L@Z;                 
                Uc = st.norm.cdf(Zc);    
                X  = np.zeros(Z.shape);       
                X  = reliability.Uc2X(Uc,dist,mean,std);  
                points = np.count_nonzero(FEL(X) < 0)
                
                nfunc=nfunc+n;     
                
                nfail=nfail+points;
                if nfail!=0:
                        Pf[k-1]    = nfail/(k*n)
                        CVPf[k-1]  = np.sqrt((1-Pf[k-1])/(k*n*Pf[k-1]))
                        beta[k-1]  =-st.norm.ppf(Pf[k-1]);         
                else:
                        Pf[k-1] = 0;CVPf[k-1]=0; beta[k-1]=0;
                
                
                        
                print('-------------------------','\n',
                      '        MONTECARLO       ','\n',
                      '-------------------------','\n',
                      'No de simulações:{0:5.0e}'.format(k*n),'\n',
                      'beta = {0:5.3f}'.format(beta[k-1]),'\n',
                      'CVPf = {0:5.0f}%'.format(CVPf[k-1]*100),'\n',
                      'Pf   = {0:5.3e}'.format(Pf[k-1]),'\n',
                      'Chamadas da FEL = {0:5.0f}'.format(nfunc),'\n')   
                
                if CVPf[k-1]<=0.05: # criteria to minimize simulations
                    break
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
