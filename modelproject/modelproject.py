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

# defining class
class SolowModelClass(): 
    
    def __init__(self,do_print=True):
            """ create the model """

            self.par = SimpleNamespace()
            self.val = SimpleNamespace()
            self.sim = SimpleNamespace()

            self.setup()

    def setup(self):
            """ baseline parameters, values and simulation vectors """
    
            val = self.val
            par = self.par
            sim = self.sim

            # model parameters for analytical solution
            par.k = sm.symbols('k')
            par.alpha = sm.symbols('alpha')
            par.delta = sm.symbols('delta')
            par.sigma =  sm.symbols('sigma')
            par.s = sm.symbols('s')
            par.g = sm.symbols('g')
            par.n = sm.symbols('n')
            par.d = sm.symbols('D')
            par.dT = sm.Symbol("\Delta T")            
            par.kss = sm.symbols('ktilde^*')
            par.yss = sm.symbols('ytilde^*')

            # model parameter values for numerical solution
            val.s = 0.3
            val.g = 0.02
            val.n = 0.01
            val.alpha = 0.33
            val.delta = 0.05
            val.sigma = 0.013258 #assuming D_100 = 0.175 and dT = 4 as in the exam
            val.d = 0
            val.dT = 4
            val.d_vec = np.linspace(0,1,10, endpoint=False)

            # simulation parameters for further analysis
            par.simT = 100 #number of periods
            sim.K = np.zeros(par.simT)
            sim.L = np.zeros(par.simT)
            sim.A = np.zeros(par.simT)
            sim.Y = np.zeros(par.simT)
            sim.fracY = np.zeros(par.simT) + np.nan
            sim.fracYD = np.zeros(par.simT)
            sim.fracYDgrowth = np.zeros(par.simT)
            sim.fracY_ext = np.zeros(par.simT)

    # analytical solution for capital in steady state
    def solve_analytical_ss(self):
        """ function that solves the model analytical and returns k in steady state

        Args:
            self :   Reference to class to call model parameters
    
        Returns: 
            kss  :     Steady state equation of capital pr capita

        """

        par = self.par

        y = (1-par.d)*par.k**par.alpha
        k_ss = sm.Eq(par.k,(par.s*y+(1-par.delta)*par.k)/((1+par.n)*(1+par.g)))
        kss = sm.solve(k_ss,par.k)[0]
        return kss

    # numerical solution for capital and output in steady state
    def solve_num_ss(self):
        """ function that numerically solves the model 
        
        Args:
            self :           Reference to class to call model parameters
    
        Returns: 
            k_ss (float):    Steady state value of capital pr capita 
            y_ss (float):    Steady state value of output pr capita

        """

        val = self.val

        y = lambda k: (1-val.d)*k**val.alpha
        obj_kss = lambda kss: kss - (val.s*y(kss) + (1-val.delta)*kss)/((1+val.g)*(1+val.n))    
        result = optimize.root_scalar(obj_kss,bracket=[0.1,100],method='brentq')

        k_ss = result.root
        y_ss = (1-val.d)*k_ss**val.alpha

        return k_ss, y_ss
    
    # solving for sigma numerically
    def solve_sigma_expression(self): 
        """ function that returning an analytical expression for sigma 
        
        Args:
            self :   Reference to class to call model parameters
    
        Returns: 
            sigma:   Equation for sigma 

        """

        par = self.par 
        eq = sm.Eq(par.d,1-(1/(1+par.sigma*(par.dT)** 2))) 
        sigma = sm.solve(eq,par.sigma)[0] 
        return sigma 
    
    # solving for sigma given t=100, D_100 = 0.175, and dT/year = 0.04  
    def solve_sigma(self): 
        """ function that numerically calculates and return a value for sigma 
        
        Args:
            self :          Reference to class to call model parameters
    
        Returns: 
            sol (float):    Value of sigma

        """

        par = self.par 
        val = self.val 
        val.d = 0.175 
        eq = sm.Eq(val.d,1-(1/(1+par.sigma*(val.dT)** 2))) 
        sol = sm.solve(eq,par.sigma)[0] 
        print(f'sigma = {sol:.6f}')

    # evaluating capital and outcome in steady state for different levels of climate damage
    def D_vector(self):
        """ function that calculates SS values for k and y as well as the value of y 
        relative to y in the baseline scenario and prints it all in a table
        
        Args:
            self :            Reference to class to call model parameters
    
        Returns: 
            k_ss_list :       List of steady state values for capita pr capita
            y_ss_list :       List of steady state values for output pr capita
            rel_y_ss_list :   List of relative output

        """

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
    
    def d_growth(self, t):
        """ function defining the damage function D
        Args:
            self    :            Reference to class to call model parameters
            t (int) :            Time variable t

        """

        val = self.val
        return 1-(1/(1+val.sigma*(0.04*t)**2))
    
    # simulating the evolution of output over a 100-year period compared to initial value
    def simulate(self, D_param):
        """ function that makes a simulation of all scenarios
         
        Args:
            self    :         Reference to class to call model parameters
            D_param :         Parameter for varying climate change damage D 
    
        Returns: 
            sim.fracY (ndarray)        :   Simulation values no climate change
            sim.fracYD (ndarray)       :   Simulation values fixed climate change
            sim.fracYDgrowth (ndarray) :   Simulation values increasing climate change

        """

        par = self.par
        val = self.val
        sim = self.sim

        # looping over each period t
        for t in range(par.simT):
            if t == 0: 
                #Setting values and equations common for all simulations in period 0. 
                K_lag = 7.235
                L_lag = 1
                A_lag = 1

                L = sim.L[t] = L_lag
                A = sim.A[t] = A_lag

                # setting the values for period 0
                if D_param == 0:
                    # setting the equations for period 0
                    Y_lag = (1-D_param)*K_lag**val.alpha*(A_lag*L_lag)**(1-val.alpha)
                    K = sim.K[t] = val.s*Y_lag+(1-val.delta)*K_lag
                    Y = sim.Y[t] = (1-D_param)*sim.K[t]**(val.alpha)*(sim.A[t]*sim.L[t])**(1-val.alpha)
                    
                    # the relative growth in GDP compared to t=0
                    sim.fracY[t] = (sim.Y[t]/sim.L[t])/(sim.Y[0]/sim.L[0])
                
                elif D_param == 0.175: 
                    # setting the equations for period 0
                    Y_lag = (1-0)*K_lag**val.alpha*(A_lag*L_lag)**(1-val.alpha)
                    K = sim.K[t] = val.s*Y_lag+(1-val.delta)*K_lag
                    Y = sim.Y[t] = (1-0)*sim.K[t]**(val.alpha)*(sim.A[t]*sim.L[t])**(1-val.alpha)
                    
                    # the relative growth in GDP compared to t=0
                    sim.fracYD[t] = (sim.Y[t]/sim.L[t])/(sim.Y[0]/sim.L[0])
                
                elif D_param == 'growth':
                    # setting the equations for period 0
                    Y_lag = (1-self.d_growth(t))*K_lag**val.alpha*(A_lag*L_lag)**(1-val.alpha)
                    K = sim.K[t] = val.s*Y_lag+(1-val.delta)*K_lag
                    Y = sim.Y[t] = (1-self.d_growth(t))*sim.K[t]**(val.alpha)*(sim.A[t]*sim.L[t])**(1-val.alpha)
                    
                    # the relative growth in GDP compared to t=0
                    sim.fracYDgrowth[t] = (sim.Y[t]/sim.L[t])/(sim.Y[0]/sim.L[0])

            else: 
            # setting values and equations common for all simulations in period t=1 to t=100. 
                K_lag = sim.K[t-1]
                L_lag = sim.L[t-1]
                A_lag = sim.A[t-1]
                Y_lag = sim.Y[t-1]
                
                L = sim.L[t] = (1+val.n)*L_lag
                A = sim.A[t] = (1+val.g)*A_lag
                K = sim.K[t] = val.s*Y_lag+(1-val.delta)*K_lag
            
                # simulation for no climate change
                if D_param == 0:
                    # output equation
                    Y = sim.Y[t] = (1-D_param)*sim.K[t]**(val.alpha)*(sim.A[t]*sim.L[t])**(1-val.alpha)
                    
                    # the relative growth in GDP compared to t=0
                    sim.fracY[t] = (sim.Y[t]/sim.L[t])/(sim.Y[0]/sim.L[0])

                # simulation for sudden climate change in periode t=1    
                elif D_param == 0.175:
                    # output equation
                    Y = sim.Y[t] = (1-D_param)*sim.K[t]**(val.alpha)*(sim.A[t]*sim.L[t])**(1-val.alpha)    
                    
                    # the relative growth in GDP compared to t=0
                    sim.fracYD[t] = (sim.Y[t]/sim.L[t])/(sim.Y[0]/sim.L[0])

                # simulation for increasing climate change
                elif D_param == 'growth': 
                    # output equation
                    Y = sim.Y[t] = (1-self.d_growth(t))*sim.K[t]**(val.alpha)*(sim.A[t]*sim.L[t])**(1-val.alpha)    
                    
                    # the relative growth in GDP compared to t=0
                    sim.fracYDgrowth[t] = (sim.Y[t]/sim.L[t])/(sim.Y[0]/sim.L[0])

    def solve_num_extension(self):
        """ solving the model numerically (with the extension)
           
        Args:
            self :            Reference to class to call model parameters
    
        Returns: 
            k_ss_ext:       List of steady state values for capita pr capita (extension)
            y_ss_ext :      List of steady state values for output pr capita (extension)

        """

        val = self.val

        f = lambda k: (1-(1-(1/(1+val.sigma*(val.dT)**2))))*k**val.alpha
        obj_kss = lambda kss_ext: kss_ext - (val.s*f(kss_ext) + (1-(val.delta+0.01*val.dT))*kss_ext)/((1+val.g)*(1+val.n))    
        result = optimize.root_scalar(obj_kss,bracket=[0.1,100],method='brentq')

        k_ss_ext = result.root
        y_ss_ext = (1-(1-(1/(1+val.sigma*(val.dT)**2))))*k_ss_ext**val.alpha

        return k_ss_ext, y_ss_ext

    def extension(self):
        """ function that simulates the model with the extension
        
        Args:
            self :            Reference to class to call model parameters
    
        Returns: 
            sim.fracY_ext:    Simulation values for extension of climate change

        """
        
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