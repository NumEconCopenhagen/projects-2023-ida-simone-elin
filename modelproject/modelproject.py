from scipy import optimize
import numpy as np
import sympy as sm
from sympy.solvers import solve
from sympy import Symbol
import matplotlib.pyplot as plt
from types import SimpleNamespace
from tabulate import tabulate

class SolowModelClass(): 
    
    def __init__(self,do_print=True):
            """ create the model """

            if do_print: print('initializing the model:')

            self.par = SimpleNamespace()
            self.val = SimpleNamespace()
            self.sim = SimpleNamespace()

            if do_print: print('calling .setup()')
            self.setup()


    def setup(self):
            """ baseline parameters """

            val = self.val
            par = self.par
            sim = self.sim

            par.k = sm.symbols('k')
            par.alpha = sm.symbols('alpha')
            par.delta = sm.symbols('delta')
            par.sigma =  sm.symbols('sigma')
            par.s = sm.symbols('s')
            par.g = sm.symbols('g')
            par.n = sm.symbols('n')
            par.d = sm.symbols('D')
            par.dT = sm.symbols('dT')

            # model parameters
            val.s = 0.3
            val.g = 0.02
            val.n = 0.01
            val.alpha = 0.33
            val.delta = 0.05
            val.sigma = 0.0132578 #fra eksamen
            val.d = 0
            val.dT = 4
            val.d_vec = np.linspace(0,1,5, endpoint=False)

            # simulation parameters
            par.simT = 100
            sim.K = np.zeros(par.simT)
            sim.L = np.zeros(par.simT)
            sim.A = np.zeros(par.simT)
            sim.Y = np.zeros(par.simT)
            sim.fracY = np.zeros(par.simT)
            sim.fracYD = np.zeros(par.simT)
            sim.fracYDgrowth = np.zeros(par.simT)


    def solve_analytical_ss(self):
        par = self.par

        f = (1-par.d)*par.k**par.alpha
        k_ss = sm.Eq(par.k,(par.s*f+(1-par.delta)*par.k)/((1+par.n)*(1+par.g)))
        kss = sm.solve(k_ss,par.k)[0]
        return kss
    
    def solve_sigma(self):
        par = self.par

        sigma_1 = sm.Eq(par.d,1-(1/(1+par.sigma*(par.dT)**2)))
        sigmavalue = sm.solve(sigma_1,par.sigma)[0]
        return sigmavalue
    

    def solve_num_ss(self):
        val = self.val

        f = lambda k: (1-val.d)*k**val.alpha
        obj_kss = lambda kss: kss - (val.s*f(kss) + (1-val.delta)*kss)/((1+val.g)*(1+val.n))    
        result = optimize.root_scalar(obj_kss,bracket=[0.1,100],method='brentq')

        k_ss = result.root
        y_ss = (1-val.d)*k_ss**val.alpha

        return k_ss, y_ss
    
    def D_vector(self):
        val = self.val

        # create an empty list to store the results
        k_ss_list = []
        y_ss_list = []
        relative_y_ss_list = []

        for d in val.d_vec:
            val.d = d
            k_ss, y_ss = self.solve_num_ss()
            k_ss_list.append(k_ss)
            y_ss_list.append(y_ss)
            rel_y_ss = y_ss/y_ss_list[0]*100
            relative_y_ss_list.append(rel_y_ss)

            #print(f'd = {d:.1f}: \n Steady state for k is {k_ss:.1f} \n steady state for y is {y_ss:.1f} \n Steady state output per worker relative to a world without climate change is {rel_y_ss:.1f}% \n')
        data = {"D": val.d_vec, "K_ss": k_ss_list, "Y_ss": y_ss_list}
        print(tabulate(data,headers="keys",tablefmt="fancy_grid"))


    def simulate(self):
        par = self.par
        val = self.val
        sim = self.sim

        #Simulating without climate change
        val.d = 0.0

        #Looping over each period t
        for t in range(par.simT):
            if t == 0: 
                #Setting the values for period 0
                K_lag = 7.235
                L_lag = 1
                A_lag = 1
                Y_lag = (1-val.d)*K_lag**val.alpha*(A_lag*L_lag)**(1-val.alpha)

                #The model equations for period 0
                L = sim.L[t] = L_lag
                A = sim.A[t] = A_lag
                K = sim.K[t] = val.s*Y_lag+(1-val.delta)*K_lag
                Y = sim.Y[t] = (1-val.d)*sim.K[t]**(val.alpha)*(sim.A[t]*sim.L[t])**(1-val.alpha)

            else: 
                #Setting the lagged values from period t=1 to t=100
                K_lag = sim.K[t-1]
                L_lag = sim.L[t-1]
                A_lag = sim.A[t-1]
                Y_lag = sim.Y[t-1]

                #The model equations for period t = 1 to t = 100
                L = sim.L[t] = (1+val.n)*L_lag
                A = sim.A[t] = (1+val.g)*A_lag
                K = sim.K[t] = val.s*Y_lag+(1-val.delta)*K_lag
                Y = sim.Y[t] = (1-val.d)*sim.K[t]**(val.alpha)*(sim.A[t]*sim.L[t])**(1-val.alpha)

            #Calculating the relative growth in GDP
            sim.fracY[t] = (sim.Y[t]/sim.L[t])/(sim.Y[0]/sim.L[0])

    def simulate2(self):
        par = self.par
        val = self.val
        sim = self.sim
      
        #Simulating with climate change from periode t = 1 
        val.d = 0.175

        #Looping over each period t
        for t in range(par.simT):
            if t == 0: 
                #Setting the values for period 0
                K_lag = 7.235
                L_lag = 1
                A_lag = 1
                Y_lag = (1-0)*K_lag**val.alpha*(A_lag*L_lag)**(1-val.alpha)

                #The model equations for period 0, with no climate change
                L = sim.L[t] = L_lag
                A = sim.A[t] = A_lag
                K = sim.K[t] = val.s*Y_lag+(1-val.delta)*K_lag
                Y = sim.Y[t] = (1-0)*sim.K[t]**(val.alpha)*(sim.A[t]*sim.L[t])**(1-val.alpha)

            else:
                #Setting the lagged values from period t=1 to t=100, with climate change from period 1
                K_lag = sim.K[t-1]
                L_lag = sim.L[t-1]
                A_lag = sim.A[t-1]
                Y_lag = sim.Y[t-1]

                #The model equations for period t = 1 to t = 100
                L = sim.L[t] = (1+val.n)*L_lag
                A = sim.A[t] = (1+val.g)*A_lag
                K = sim.K[t] = val.s*Y_lag+(1-val.delta)*K_lag
                Y = sim.Y[t] = (1-val.d)*sim.K[t]**(val.alpha)*(sim.A[t]*sim.L[t])**(1-val.alpha)
            
            #Calculating the relative growth in GDP with climate change  
            sim.fracYD[t] = (sim.Y[t]/sim.L[t])/(sim.Y[0]/sim.L[0])

    def simulate3(self):
        par = self.par
        val = self.val
        sim = self.sim

        #Looping over each period t
        for t in range(par.simT):
            
            #The model equation for D 
            def d_growth(self):
                return 1-(1/(1+val.sigma*(0.04*t)**2))
            
            if t == 0: 
                #Setting the values for period 0
                K_lag = 7.235
                L_lag = 1
                A_lag = 1
                Y_lag = (1-d_growth(self))*K_lag**val.alpha*(A_lag*L_lag)**(1-val.alpha)
                
                #The model equations for period 0
                L = sim.L[t] = L_lag
                A = sim.A[t] = A_lag
                K = sim.K[t] = val.s*Y_lag+(1-val.delta)*K_lag
                Y = sim.Y[t] = (1-d_growth(self))*sim.K[t]**(val.alpha)*(sim.A[t]*sim.L[t])**(1-val.alpha)
            
            else:
                #Setting the lagged values from period t=1 to t=100
                K_lag = sim.K[t-1]
                L_lag = sim.L[t-1]
                A_lag = sim.A[t-1]
                Y_lag = sim.Y[t-1]

                #The model equations for period t = 1 to t = 100
                L = sim.L[t] = (1+val.n)*L_lag
                A = sim.A[t] = (1+val.g)*A_lag
                K = sim.K[t] = val.s*Y_lag+(1-val.delta)*K_lag
                Y = sim.Y[t] = (1-d_growth(self))*sim.K[t]**(val.alpha)*(sim.A[t]*sim.L[t])**(1-val.alpha)
            
            #Calculating the relative growth in GDP with growing climate change 
            sim.fracYDgrowth[t] = (sim.Y[t]/sim.L[t])/(sim.Y[0]/sim.L[0])





    def simulatesamlet(self):
        par = self.par
        val = self.val
        sim = self.sim
        # period-by-period
        for t in range(par.simT) :  
            def d_growth(self):
                return 1-(1/(1+val.sigma*(0.04*t)**2))
              
            d_list = [0,0.175,d_growth(self)]
            for d in d_list : 
                if t == 0: 
                    K_lag = 7.235
                    L_lag = 1
                    A_lag = 1
                    Y_lag = (1-d_list[d])*K_lag**val.alpha*(A_lag*L_lag)**(1-val.alpha)

                    L = sim.L[t] = L_lag
                    A = sim.A[t] = A_lag
                    K = sim.K[t] = val.s*Y_lag+(1-val.delta)*K_lag
                    Y = sim.Y[t] = (1-d_list[d])*sim.K[t]**(val.alpha)*(sim.A[t]*sim.L[t])**(1-val.alpha)

                else:
                    K_lag = sim.K[t-1]
                    L_lag = sim.L[t-1]
                    A_lag = sim.A[t-1]
                    Y_lag = sim.Y[t-1]

                    L = sim.L[t] = (1+val.n)*L_lag
                    A = sim.A[t] = (1+val.g)*A_lag
                    K = sim.K[t] = val.s*Y_lag+(1-val.delta)*K_lag
                    Y = sim.Y[t] = (1-d_list[d])*sim.K[t]**(val.alpha)*(sim.A[t]*sim.L[t])**(1-val.alpha)

                sim.fracY[t] = (sim.Y[t]/sim.L[t])/(sim.Y[0]/sim.L[0])
                sim.fracYD[t]= (sim.Y[t]/sim.L[t])/(sim.Y[0]/sim.L[0])
                sim.fracYDgrowth[t] = (sim.Y[t]/sim.L[t])/(sim.Y[0]/sim.L[0])