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

            par.k = sm.symbols('k')
            par.alpha = sm.symbols('alpha')
            par.delta = sm.symbols('delta')
            par.sigma =  sm.symbols('sigma')
            par.s = sm.symbols('s')
            par.g = sm.symbols('g')
            par.n = sm.symbols('n')
            par.d = sm.symbols('D')

            #model parameters
            val.s = 0.2
            val.g = 0.02
            val.n = 0.01
            val.alpha = 1/3
            val.delta = 0.1
            val.sigma = 1/2
            val.d = 1/2


    def solve_analytical_ss(self):
        par = self.par

        f = (1-par.d)*par.k**par.alpha
        k_ss = sm.Eq(par.k,(par.s*f+(1-par.delta)*par.k)/((1+par.n)*(1+par.g)))
        kss = sm.solve(k_ss,par.k)[0]
        return kss

    def solve_ss(self):
        par = self.par

        f = lambda k: (1-par.d)*par.k**par.alpha
        obj_kss = lambda kss: kss - (par.s*f(kss) + (1-par.delta)*kss)/((1+par.g)*(1+par.n))    
        result = optimize.root_scalar(obj_kss,bracket=[0.1,100],method='brentq')

        return result 

    def :
    ss_func = sm.lambdify((s,g,n,delta,alpha,sigma,d),kss)

    # Evaluate function
    ss_func(0.2,0.02,0.01,0.1,1/3,1/2,1/2)
