from scipy import optimize
import numpy as np
import sympy as sm
import matplotlib.pyplot as plt
from types import SimpleNamespace


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

            # model parameters
            val.s = 0.2
            val.g = 0.02
            val.n = 0.01
            val.alpha = 0.33
            val.delta = 0.1
            val.sigma = 0.5
            val.d = 0.5
            #val.d_vec = np.linspace(0,1,5, endpoint=False)

            # simulation parameters
            par.simT = 100

            sim.s = np.zeros(par.simT)
            sim.g = np.zeros(par.simT)
            sim.n = np.zeros(par.simT)
            sim.alpha = np.zeros(par.simT)
            sim.delta = np.zeros(par.simT)
            sim.sigma = np.zeros(par.simT)
            sim.d = np.zeros(par.simT)
            sim.K = np.zeros(par.simT)
            sim.L = np.zeros(par.simT)
            sim.A = np.zeros(par.simT)
            sim.Y = np.zeros(par.simT)
            sim.fracY = np.zeros(par.simT)


    def solve_analytical_ss(self):
        par = self.par

        f = (1-par.d)*par.k**par.alpha
        k_ss = sm.Eq(par.k,(par.s*f+(1-par.delta)*par.k)/((1+par.n)*(1+par.g)))
        kss = sm.solve(k_ss,par.k)[0]
        return kss

    def solve_num_ss(self):
        val = self.val

        f = lambda k: (1-val.d)*k**val.alpha
        obj_kss = lambda kss: kss - (val.s*f(kss) + (1-val.delta)*kss)/((1+val.g)*(1+val.n))    
        result = optimize.root_scalar(obj_kss,bracket=[0.1,100],method='brentq')

        k_ss = result.root
        y_ss = (1-val.d)*k_ss**val.alpha

        return k_ss, y_ss
    

    def simulate(self):
        par = self.par
        val = self.val
        sim = self.sim

         # period-by-period
        for t in range(par.simT):

            if t == 0: 
                K_lag = 7.235
                L_lag = 1
                A_lag = 1
                Y_lag = (1-val.d)*K_lag**val.alpha*(A_lag*L_lag)**(1-val.alpha)

            else:
                K_lag = sim.K[t-1]
                L_lag = sim.L[t-1]
                A_lag = sim.A[t-1]
                Y_lag = sim.Y[t-1]

            
            L = sim.L[t] = (1+val.n)*L_lag
            A = sim.A[t] = (1+val.g)*A_lag
            K = sim.K[t] = val.s*sim.Y[t-1]+(1-val.delta)*K_lag
            Y = sim.Y[t] = (1-val.d)*sim.K[t]**(val.alpha)*(sim.A[t]+sim.L[t])**(1-val.alpha)

            fracY = sim.fracY[t] = (sim.Y[t]/Y)

