# import packages
from types import SimpleNamespace

import numpy as np

from scipy import optimize

from scipy.optimize import minimize

import pandas as pd 

import matplotlib.pyplot as plt
from matplotlib import ticker

# setup household class
class HouseholdClass:

    def __init__(self):
        """ setup model """

        # create namespaces
        par = self.par = SimpleNamespace()
        sol = self.sol = SimpleNamespace()

        # preferences
        par.rho = 2.0
        par.nu = 0.001
        par.epsilon = 1.0
        par.omega = 0.5 

        # household production
        par.alpha = 0.5
        par.sigma = 1

        # wages
        par.wM = 1.0
        par.wF = 1.0
        par.wF_vec = np.linspace(0.8,1.2,5)

        # targets
        par.beta0_target = 0.4
        par.beta1_target = -0.1

        # solution
        sol.LM_vec = np.zeros(par.wF_vec.size)
        sol.HM_vec = np.zeros(par.wF_vec.size)
        sol.LF_vec = np.zeros(par.wF_vec.size)
        sol.HF_vec = np.zeros(par.wF_vec.size)

        #Empty solution vectors for extention in question 5
        sol.LM_vec_ext = np.zeros(par.wF_vec.size)
        sol.HM_vec_ext = np.zeros(par.wF_vec.size)
        sol.LF_vec_ext = np.zeros(par.wF_vec.size)
        sol.HF_vec_ext = np.zeros(par.wF_vec.size)

        sol.beta0 = np.nan
        sol.beta1 = np.nan

    def calc_utility(self,LM,HM,LF,HF):
        """ calculate utility """

        # unpack 
        par = self.par
        sol = self.sol

        # consumption of market goods
        C = par.wM*LM + par.wF*LF

        # consumption of home goods
        H = None 
        if (par.sigma == 0):
            H = min(HM,HF)
        elif (par.sigma == 1):
            H = HM**(1-par.alpha)*HF**par.alpha
        else:
            H = ((1-par.alpha) * HM**((par.sigma-1)/par.sigma) + par.alpha * HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))

        # total consumption utility
        Q = C**(par.omega)*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # disutility of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        # total utility
        return utility - disutility

    def solve_discrete(self,do_print=False):
        """ solve model discretely """
        
        # unpack
        par = self.par
        sol = self.sol
        opt = SimpleNamespace()
        
        # all possible choices
        x = np.linspace(0,24,49)
        LM,HM,LF,HF = np.meshgrid(x,x,x,x) # all combinations
    
        LM = LM.ravel() # vector
        HM = HM.ravel()
        LF = LF.ravel()
        HF = HF.ravel()

        # calculate utility
        u = self.calc_utility(LM,HM,LF,HF)
    
        # set to minus infinity if constraint is broken
        I = (LM+HM > 24) | (LF+HF > 24) # | is "or"
        u[I] = -np.inf
    
        # find maximizing argument
        j = np.argmax(u)
        
        opt.LM = LM[j]
        opt.HM = HM[j]
        opt.LF = LF[j]
        opt.HF = HF[j]

        # print
        if do_print:
            for k,v in opt.__dict__.items():
                print(f'{k} = {v:6.4f}')

        return opt
   
    def solve_obj(self, x, do_print=False):
            """ fetching utility from class"""
            value = self.calc_utility(x[0],x[1],x[2],x[3])

            # we perform a monotone transformation by multiplying with 100 to avoid numerical inaccuracy for small numbers
            return - value*100


    def solve_continuous(self,do_print=False):
        """ solve model continously """
    
        # unpack
        par = self.par
        sol = self.solcont = SimpleNamespace()

        # constraints
        constraints = [{'type': 'ineq', 'fun': lambda x: 24-(x[0]+x[1])},
                    {'type': 'ineq', 'fun': lambda x: 24-(x[2]+x[3])}]

        # guesses
        guess = [12]*4

        # bounds
        bounds = [(0,24)]*4

        # optimizer 
        result = minimize(self.solve_obj,guess,method='SLSQP', bounds=bounds, constraints=constraints) 


        # results
        sol.LM=result.x[0]
        sol.HM=result.x[1]
        sol.LF=result.x[2]
        sol.HF=result.x[3]

        # return solution
        return sol
        
    #Solving our model for each iteration of female wage (used for both discrete and continuous model)
    def solve_wF_vec(self,discrete=False):
        """ solve model for vector of female wages """
        par = self.par
        sol = self.sol
        for it, wF in enumerate(par.wF_vec):
            par.wF = wF
            if discrete == True:
                res = self.solve_discrete()
            else:
                res = self.solve_continuous()
            sol.HF_vec[it] = res.HF
            sol.HM_vec[it] = res.HM
            sol.LM_vec[it] = res.LM
            sol.LF_vec[it] = res.LF     

    #Run the regression from the article on our data
    def run_regression(self):
        """ run regression """

        par = self.par
        sol = self.sol

        x = np.log(par.wF_vec)
        y = np.log(sol.HF_vec/sol.HM_vec)
        A = np.vstack([np.ones(x.size),x]).T
        sol.beta0,sol.beta1 = np.linalg.lstsq(A,y,rcond=None)[0]
    
   
    #Minimizing the distance between our model's beta values and the article's beta values by choosing appropriate sigma and beta
    def estimate(self,alpha=None,sigma=None):
        """ estimate alpha and sigma """

        #Unpacking
        sol = self.sol
        par = self.par

        #Objective function
        def objective(x, self):
            par.alpha = x[0]
            par.sigma = x[1]
            self.solve_wF_vec()
            self.run_regression()

            return (0.4-sol.beta0)**2+(-0.1-sol.beta1)**2

        # guesses
        guess_estimate = np.array([0.5, 0.5])

        # bounds
        bounds = [(0,10), (0,10)]

        # optimize
        optimal_result = optimize.minimize(objective, guess_estimate, args = (self), method = 'Nelder-Mead', bounds=bounds)
        
    
        # results
        sol.alpha = optimal_result.x[0]
        sol.sigma = optimal_result.x[1]


        # return solution
        return sol
    
    def solve_obj_ext(self, x, do_print=False):
            """ fetching utility from class"""
            value = self.calc_utility_ext(x[0],x[1],x[2],x[3])
            return - value*100

    def calc_utility_ext(self,LM,HM,LF,HF):
        """ calculate utility """

        # unpack 
        par = self.par
        sol = self.sol

        # consumption of market goods
        C = par.wM*LM + par.wF*LF

        # consumption of home goods
        H = None 
        if (par.sigma == 0):
            H = min(HM,HF)
        elif (par.sigma == 1):
            H = HM**(1-par.alpha)*HF**par.alpha
        else:
            H = ((1-par.alpha) * HM**((par.sigma-1)/par.sigma) + par.alpha * HF**((par.sigma-1)/par.sigma))**(par.sigma/(par.sigma-1))

        # total consumption utility
        Q = C**(par.omega)*H**(1-par.omega)
        utility = np.fmax(Q,1e-8)**(1-par.rho)/(1-par.rho)

        # disutility of work
        epsilon_ = 1+1/par.epsilon
        TM = LM+0.5*HM
        TF = LF+HF
        disutility = par.nu*(TM**epsilon_/epsilon_+TF**epsilon_/epsilon_)
        
        # total utility
        return utility - disutility

    def solve_continuous_ext(self,do_print=False):
        """ solve model continously """
    
        # unpack
        par = self.par
        sol = self.solcont = SimpleNamespace()

        # constraints
        constraints = [{'type': 'ineq', 'fun': lambda x: 24-(x[0]+0.5*x[1])},
                    {'type': 'ineq', 'fun': lambda x: 24-(x[2]+x[3])}]
    
        # guesses
        guess = [12]*4

        # bounds
        bounds = [(0,24)]*4

        # optimizer 
        result_ext = minimize(self.solve_obj_ext,guess,method='SLSQP', bounds=bounds, constraints=constraints) 


        # results
        sol.LM_vec_ext=result_ext.x[0]
        sol.HM_vec_ext=result_ext.x[1]
        sol.LF_vec_ext=result_ext.x[2]
        sol.HF_vec_ext=result_ext.x[3]

        # return solution
        return sol
    
    def solve_wF_vec_ext(self):
        """ solve extended model for vector of female wages """
        par = self.par
        sol = self.sol
        for it, wF in enumerate(par.wF_vec):
            par.wF = wF
            res = self.solve_continuous_ext()

            sol.HF_vec_ext[it] = res.HF_vec_ext
            sol.HM_vec_ext[it] = res.HM_vec_ext
            sol.LM_vec_ext[it] = res.LM_vec_ext
            sol.LF_vec_ext[it] = res.LF_vec_ext
    
def figure(x,y,title) :
    """ plot for figures in question 2 and 3"""
    fig,ax = plt.subplots(figsize=(6,5))
    ax.scatter(x,y,s=50)
    ax.set_xlabel(r"$log(\frac{wF}{wM})$",fontsize=14)
    ax.set_ylabel(r"$log(\frac{HF}{HM})$",fontsize=14)          
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.2f}"))
    ax.set_title(title,pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid()