import numpy as np
import matplotlib.pyplot as plt

from parameters import *
from models import *

def simulate_stochastic_toggle(params, Y0, Omega, T_end): 
            
        state = np.array(Y0)
        
        Y_total = []
        Y_total.append(state)
        t = 0 
        T = [] 
        T.append(t) 
   
        N = toggle_generate_stoichiometry()
        
        while t < T_end: 
            if t < T_end/5:
                params[-4] = 0 # rho_x
                params[-3] = 0 # rho_y
            elif t < 2*T_end/5:
                params[-4] = 5 # rho_x
                params[-3] = 0 # rho_y
            elif t < 3*T_end/5:
                params[-4] = 0 # rho_x
                params[-3] = 0 # rho_y
            elif t < 4*T_end/5:
                params[-4] = 0 # rho_x
                params[-3] = 5 # rho_y
            else:
                params[-4] = 0 # rho_x
                params[-3] = 0 # rho_y
            



            #choose two random numbers 
            r = np.random.uniform(size=2)
            r1 = r[0]
            r2 = r[1]           

            a = toggle_model_stochastic(state, params, Omega)
                    
            asum = np.cumsum(a)
            a0 = np.sum(a)  
            #get tau
            tau = (1.0/a0)*np.log(1.0/r1)    
        
            #print(t)       
            #select reaction 
            reaction_number = np.argwhere(asum > r2*a0)[0,0] #get first element         
        
            #update concentrations
            state = state + N[:,reaction_number]      
            Y_total.append(state) 
            #update time
            t = t + tau  
            T.append(t)

            
        T = np.array(T)
        Y_total = np.array(Y_total) 
        return T, Y_total     


#params = delta_L, gamma_A, gamma_B, n_a, n_b, theta_A, theta_B, eta_a, eta_b, omega_a, omega_b, m_a, m_b, delta_a, delta_b, rho_a, rho_b, r_A, r_B
params = [delta_L, gamma_B, gamma_B, n_b, n_b, theta_B, theta_B, eta_b, eta_b, omega_b, omega_b, m_b, m_b, delta_b, delta_b, rho_b, rho_b, r_B, r_B]
# reaction space volume for the whole cell population
# N_cells should be set to 1
Omega = 10


#Y = L_A, L_B, a, b, N_A, N_B
Y0 = np.zeros(6)#([0.0]*6)
# number of cells
Y0[-2] = 1
Y0[-1] = 1

#Y0[0] = 0.1 # L_A
Y0[1] = 1 # L_B
Y0[2] = 1 # a
#Y0[3] = 0.1 # b

T, Y = simulate_stochastic_toggle(params, Y0, Omega, 500)


L_A = Y[:,0]
L_B = Y[:,1]
a = Y[:,2]
b = Y[:,3]
N_A = Y[:,4]
N_B = Y[:,5]

ax1 = plt.subplot(311)
ax1.plot(T,L_A)
ax1.plot(T,L_B)
ax1.legend(["L_A", "L_B"])

ax2 = plt.subplot(312)
ax2.plot(T,a)
ax2.plot(T,b)
ax2.legend(["a", "b"])

ax3 = plt.subplot(313)
ax3.plot(T,N_A)
ax3.plot(T,N_B)
ax3.legend(["N_A", "N_B"])

#plt.plot(Y)
#plt.legend(["L_A", "L_B", "a", "b", "N_A", "N_B"])
#plt.xticks([0, len(T)-1], [0, T[-1]])

plt.show()
