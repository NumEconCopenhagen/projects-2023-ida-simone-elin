from scipy import optimize
import numpy as np
import sympy as sm
from sympy.solvers import solve
from sympy import Symbol
import matplotlib.pyplot as plt
from types import SimpleNamespace
from tabulate import tabulate
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual

#Defining class
class SolowModelClass(): 
    
    def __init__(self,do_print=True):
            """ create the model """

            # if do_print: print('initializing the model:')

            self.par = SimpleNamespace()
            self.val = SimpleNamespace()
            self.sim = SimpleNamespace()

            # if do_print: print('calling .setup()')
            self.setup()

    def setup(self):
            """ baseline parameters """
    
            val = self.val
            par = self.par
            sim = self.sim

            #model parameters for analytical solution
            par.k = sm.symbols('k')
            par.alpha = sm.symbols('alpha')
            par.delta = sm.symbols('delta')
            par.sigma =  sm.symbols('sigma')
            par.s = sm.symbols('s')
            par.g = sm.symbols('g')
            par.n = sm.symbols('n')
            par.d = sm.symbols('D')
            par.dT = sm.symbols('dT') #change in temperature
            par.kss = sm.symbols(r'$\tilde k_t$')

            # model parameter values for numerical solution
            val.s = 0.3
            val.g = 0.02
            val.n = 0.01
            val.alpha = 0.33
            val.delta = 0.05
            val.sigma = 0.013258 #assuming D_100 = 0.175 and dT = 4 
            val.d = 0
            val.dT = 4
            val.d_vec = np.linspace(0,1,10, endpoint=False)

            # simulation parameters for further analysis
            par.simT = 100 #number of periods
            sim.K = np.zeros(par.simT)
            sim.L = np.zeros(par.simT)
            sim.A = np.zeros(par.simT)
            sim.Y = np.zeros(par.simT)
            sim.fracY = np.zeros(par.simT)
            sim.fracYD = np.zeros(par.simT)
            sim.fracYDgrowth = np.zeros(par.simT)
            sim.fracY_ext = np.zeros(par.simT)

    # analytical solution for capital in steady state
    def solve_analytical_ss(self):
        par = self.par

        f = (1-par.d)*par.k**par.alpha
        k_ss = sm.Eq(par.k,(par.s*f+(1-par.delta)*par.k)/((1+par.n)*(1+par.g)))
        kss = sm.solve(k_ss,par.k)[0]
        return kss
    
    # solving for sigma numerically
    def solve_sigma_expression(self): 
        par = self.par 
        eq = sm.Eq(par.d,1-(1/(1+par.sigma*(par.dT)** 2))) 
        sigma = sm.solve(eq,par.sigma)[0] 
        return sigma 
    
    # solving for sigma given t=100, D_100 = 0.175, and dT/year = 0.04  
    def solve_sigma(self): 
        par = self.par 
        val = self.val 
        val.d = 0.175 
        eq = sm.Eq(val.d,1-(1/(1+par.sigma*(val.dT)** 2))) 
        sol = sm.solve(eq,par.sigma)[0] 
        print(f'sigma = {sol:.6f}')

    # numerical solution for capital and output in steady state
    def solve_num_ss(self):
        val = self.val

        f = lambda k: (1-val.d)*k**val.alpha
        obj_kss = lambda kss: kss - (val.s*f(kss) + (1-val.delta)*kss)/((1+val.g)*(1+val.n))    
        result = optimize.root_scalar(obj_kss,bracket=[0.1,100],method='brentq')

        k_ss = result.root
        y_ss = (1-val.d)*k_ss**val.alpha

        return k_ss, y_ss
    
    # evaluating capital and outcome in steady state for different levels of climate damage
    def D_vector(self):
        val = self.val

        # create an empty list to store the results
        k_ss_list = []
        y_ss_list = []
        relative_y_ss_list = []

        # saving results while looping through d-values
        for d in val.d_vec:
            val.d = d
            k_ss, y_ss = self.solve_num_ss()
            k_ss_list.append(k_ss)
            y_ss_list.append(y_ss)
            rel_y_ss = y_ss/y_ss_list[0]*100
            relative_y_ss_list.append(rel_y_ss)

        # printing results in table
        data = {"D": val.d_vec, "K_ss": k_ss_list, "Y_ss": y_ss_list, "Relative Y_ss compared \n to situation where \n D = 0 (in pct.)": relative_y_ss_list}
        print(tabulate(data,headers="keys",tablefmt="fancy_grid"))

    # simulating the evolution of output over a 100-year period compared to initial value
    def simulate(self):
        par = self.par
        val = self.val
        sim = self.sim

        # simulating without climate change
        val.d = 0.0

        # looping over each period t
        for t in range(par.simT):
            if t == 0: 
                # setting the values for period 0
                K_lag = 7.235
                L_lag = 1
                A_lag = 1
                Y_lag = (1-val.d)*K_lag**val.alpha*(A_lag*L_lag)**(1-val.alpha)

                # the model equations for period 0
                L = sim.L[t] = L_lag
                A = sim.A[t] = A_lag
                K = sim.K[t] = val.s*Y_lag+(1-val.delta)*K_lag
                Y = sim.Y[t] = (1-val.d)*sim.K[t]**(val.alpha)*(sim.A[t]*sim.L[t])**(1-val.alpha)

            else: 
                # setting the lagged values from period t=1 to t=100
                K_lag = sim.K[t-1]
                L_lag = sim.L[t-1]
                A_lag = sim.A[t-1]
                Y_lag = sim.Y[t-1]

                # the model equations for period t = 1 to t = 100
                L = sim.L[t] = (1+val.n)*L_lag
                A = sim.A[t] = (1+val.g)*A_lag
                K = sim.K[t] = val.s*Y_lag+(1-val.delta)*K_lag
                Y = sim.Y[t] = (1-val.d)*sim.K[t]**(val.alpha)*(sim.A[t]*sim.L[t])**(1-val.alpha)

            # calculating the relative growth in GDP
            sim.fracY[t] = (sim.Y[t]/sim.L[t])/(sim.Y[0]/sim.L[0])

    # simulating the evolution of output over a 100-year period compared to initial value
    def simulate2(self):
        par = self.par
        val = self.val
        sim = self.sim
      
        # simulating with climate change damages of 17.5% from periode t = 1 
        val.d = 0.175

        # looping over each period t
        for t in range(par.simT):
            if t == 0: 
                # setting the values for period 0
                K_lag = 7.235
                L_lag = 1
                A_lag = 1
                Y_lag = (1-0)*K_lag**val.alpha*(A_lag*L_lag)**(1-val.alpha)

                # the model equations for period 0, with no climate change
                L = sim.L[t] = L_lag
                A = sim.A[t] = A_lag
                K = sim.K[t] = val.s*Y_lag+(1-val.delta)*K_lag
                Y = sim.Y[t] = (1-0)*sim.K[t]**(val.alpha)*(sim.A[t]*sim.L[t])**(1-val.alpha)

            else:
                # setting the lagged values from period t = 1 to t = 100, with climate change from period 1
                K_lag = sim.K[t-1]
                L_lag = sim.L[t-1]
                A_lag = sim.A[t-1]
                Y_lag = sim.Y[t-1]

                # the model equations for period t = 1 to t = 100
                L = sim.L[t] = (1+val.n)*L_lag
                A = sim.A[t] = (1+val.g)*A_lag
                K = sim.K[t] = val.s*Y_lag+(1-val.delta)*K_lag
                Y = sim.Y[t] = (1-val.d)*sim.K[t]**(val.alpha)*(sim.A[t]*sim.L[t])**(1-val.alpha)
            
            # calculating the relative growth in GDP with climate change  
            sim.fracYD[t] = (sim.Y[t]/sim.L[t])/(sim.Y[0]/sim.L[0])
    
    # simulating the evolution of output over a 100-year period compared to initial value
    def simulate3(self):
        par = self.par
        val = self.val
        sim = self.sim

        # looping over each period t
        for t in range(par.simT):
            
            # simulating with climate change damages increasing by 0.04 degrees Celcius per year
            def d_growth(self):
                return 1-(1/(1+val.sigma*(0.04*t)**2))
            
            if t == 0: 
                # setting the values for period 0
                K_lag = 7.235
                L_lag = 1
                A_lag = 1
                Y_lag = (1-d_growth(self))*K_lag**val.alpha*(A_lag*L_lag)**(1-val.alpha)
                
                # the model equations for period 0
                L = sim.L[t] = L_lag
                A = sim.A[t] = A_lag
                K = sim.K[t] = val.s*Y_lag+(1-val.delta)*K_lag
                Y = sim.Y[t] = (1-d_growth(self))*sim.K[t]**(val.alpha)*(sim.A[t]*sim.L[t])**(1-val.alpha)
            
            else:
                # setting the lagged values from period t = 1 to t = 100
                K_lag = sim.K[t-1]
                L_lag = sim.L[t-1]
                A_lag = sim.A[t-1]
                Y_lag = sim.Y[t-1]

                # the model equations for period t = 1 to t = 100
                L = sim.L[t] = (1+val.n)*L_lag
                A = sim.A[t] = (1+val.g)*A_lag
                K = sim.K[t] = val.s*Y_lag+(1-val.delta)*K_lag
                Y = sim.Y[t] = (1-d_growth(self))*sim.K[t]**(val.alpha)*(sim.A[t]*sim.L[t])**(1-val.alpha)
            
            # calculating the relative growth in GDP with growing climate change 
            sim.fracYDgrowth[t] = (sim.Y[t]/sim.L[t])/(sim.Y[0]/sim.L[0])


    def solve_num_extension(self):
        val = self.val

        f = lambda k: (1-(1-(1/(1+val.sigma*(val.dT)**2))))*k**val.alpha
        obj_kss = lambda kss_ext: kss_ext - (val.s*f(kss_ext) + (1-(val.delta+0.01*val.dT))*kss_ext)/((1+val.g)*(1+val.n))    
        result = optimize.root_scalar(obj_kss,bracket=[0.1,100],method='brentq')

        k_ss_ext = result.root
        y_ss_ext = (1-(1-(1/(1+val.sigma*(val.dT)**2))))*k_ss_ext**val.alpha

        return k_ss_ext, y_ss_ext

    def extension(self):
        par = self.par
        val = self.val
        sim = self.sim

        # looping over each period t
        for t in range(par.simT):

            val.delta_ext = val.delta+0.01*0.04*t
            
            # simulating with climate change damages increasing by 0.04 degrees Celcius per year
            def d_growth(self):
                return 1-(1/(1+val.sigma*(0.04*t)**2))
            
            if t == 0: 
                # setting the values for period 0
                K_lag = 7.235 #from exam 
                L_lag = 1
                A_lag = 1
                Y_lag = (1-d_growth(self))*K_lag**val.alpha*(A_lag*L_lag)**(1-val.alpha)
                
                # the model equations for period 0
                L = sim.L[t] = L_lag
                A = sim.A[t] = A_lag
                K = sim.K[t] = val.s*Y_lag+(1-val.delta_ext)*K_lag
                Y = sim.Y[t] = (1-d_growth(self))*sim.K[t]**(val.alpha)*(sim.A[t]*sim.L[t])**(1-val.alpha)
            
            else:
                # setting the lagged values from period t = 1 to t = 100
                K_lag = sim.K[t-1]
                L_lag = sim.L[t-1]
                A_lag = sim.A[t-1]
                Y_lag = sim.Y[t-1]

                # the model equations for period t = 1 to t = 100
                L = sim.L[t] = (1+val.n)*L_lag
                A = sim.A[t] = (1+val.g)*A_lag
                K = sim.K[t] = val.s*Y_lag+(1-val.delta_ext)*K_lag
                Y = sim.Y[t] = (1-d_growth(self))*sim.K[t]**(val.alpha)*(sim.A[t]*sim.L[t])**(1-val.alpha)
            
            # calculating the relative growth in GDP with growing climate change 
            sim.fracY_ext[t] = (sim.Y[t]/sim.L[t])/(sim.Y[0]/sim.L[0])