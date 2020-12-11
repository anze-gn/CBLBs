import numpy as np
import matplotlib.pyplot as plt

from parameters import *
from models import *

def simulate_stochastic_clb(params, Y0, Omega, T_end, dt = 1): 
            
        state = np.array(Y0)
        
        Y_total = np.zeros([1+T_end//dt, len(state)])
        T = np.zeros(1+T_end//dt)
        t = 0 

        Y_total[0, :] = state
        T[0] = t
   
        N = CLB_generate_stoichiometry()

        i = 1
        last_time = t

        while t < T_end: 
            
            """
            if t < T_end/3:
                rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b = 0, 5, 5, 0, 5, 0, 5, 0
            elif t < 2*T_end/3:
                rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b = 0, 0, 0, 0, 0, 0, 0, 0
            else:
                rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b = 5, 0, 0, 5, 0, 0, 0, 0

            params[-8:] = rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b

            if t > T_end/2:
                S = np.array([1, 0])
                state[24:26] = S*Omega
            """
            if t > T_end/2:
                #params[-8:] = rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b
                params[-8:] = 0, 0, 0, 0, 0, 0, 0, 0
              

            #choose two random numbers 
            r = np.random.uniform(size=2)
            r1 = r[0]
            r2 = r[1]           

            a = CLB_model_stochastic(state, params, Omega)
                    
            asum = np.cumsum(a)
            a0 = np.sum(a)  
            #get tau
            tau = (1.0/a0)*np.log(1.0/r1)    
        
            #print(t)       
            #select reaction 
            reaction_number = np.argwhere(asum > r2*a0)[0,0] #get first element         
        
            #update concentrations
            state = state + N[:,reaction_number]      
            
            #update time
            t = t + tau  
   

            if (t - last_time >= dt) or (t >= T_end):
                last_time = t
                Y_total[i, :] = state
                T[i] = t                
                i += 1
            


        return T[:i], Y_total[:i,:]     

Y0 = np.zeros(59)


# number of cells: toggle switches
N_I0 = np.array([1,1])
N_I1 = np.array([1,1])
N_I2 = np.array([1,1])
N_I3 = np.array([1,1])

Y0[4:6] = N_I0
Y0[10:12] = N_I1
Y0[16:18] = N_I2
Y0[22:24] = N_I3

# number of cells: mux
#Y0[22-4+24:38-4+24] = 1 # number of cells
Y0[42:58] = 1 # number of cells


# reaction space volume for a single cell
Omega = 10

t_end = 500


states = [([0,0], [0,0,0,0]), 
          ([0,0], [1,0,0,0]), 
          ([1,0], [1,0,0,0]),
          ([1,0], [0,1,0,0]), 
          ([0,1], [0,1,0,0]),
          ([0,1], [0,0,1,0]), 
          ([1,1], [0,0,1,0]), 
          ([1,1], [0,0,0,1])]
"""
states = [([0,0], [0,0,0,0]), ([0,0], [1,0,0,0]), 
          ([1,0], [1,0,0,0]), ([1,0], [1,1,0,0]), 
          ([0,1], [1,1,0,0]), ([0,1], [1,1,1,0]), 
          ([1,1], [1,1,1,0]), ([1,1], [1,1,1,1])]
"""

for iteration, state in enumerate(states):
    S = state[0]
    I = state[1]
    I0, I1, I2, I3 = I

    rho_x = 0
    rho_y = 0
    
    
    if iteration > 0 and states[iteration-1][1] == I:
        #rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b = (1-I0) * 5, I0*5, (1-I1)*5, I1*5, (1-I2)*5, I2*5, (1-I3)*5, I3*5
        rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b = 0, 0, 0, 0, 0, 0, 0, 0        
    else:
        rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b = (1-I0) * 5, I0*5, (1-I1)*5, I1*5, (1-I2)*5, I2*5, (1-I3)*5, I3*5
    #rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b = 5, 0, 5, 0, 5, 0, 5, 0
    
    params = [delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, delta_y, rho_x, rho_y, gamma_x, theta_x, r_X, r_Y, 
             rho_I0_a, rho_I0_b, rho_I1_a, rho_I1_b, rho_I2_a, rho_I2_b, rho_I3_a, rho_I3_b]
    
        

    if iteration:
        Y0 = Y_full[-1,:]        
    else:
        Y0 *= Omega

    #print(Y0)

    Y0[24:26] = np.array(S) * Omega

    T, Y = simulate_stochastic_clb(params, Y0, Omega, t_end)

    if not iteration:
        Y_full = Y
        T_full = T
    else:        
        Y_full = np.append(Y_full, Y, axis = 0)
        T_full = np.append(T_full, T + T_full[-1], axis = 0)

Y = Y_full
T = T_full


"""
results
"""


out = Y[:,-1]

S0, S1 = Y[:,24], Y[:,25]

I0_a, I0_b = Y[:,2], Y[:,3]
I1_a, I1_b = Y[:,8], Y[:,9]
I2_a, I2_b = Y[:,14], Y[:,15]
I3_a, I3_b = Y[:,20], Y[:,21]

# plot
"""
ax1 = plt.subplot(241)
ax1.plot(T, I0_a)
ax1.plot(T, I0_b)
ax1.legend(["I0_a = I0", "I0_b"])
ax1.set_title('I0 toggle')

ax2 = plt.subplot(242)
ax2.plot(T, I1_a)
ax2.plot(T, I1_b)
ax2.legend(["I1_a = I1", "I1_b"])
ax2.set_title('I1 toggle')

ax3 = plt.subplot(243)
ax3.plot(T, I2_a)
ax3.plot(T, I2_b)
ax3.legend(["I2_a = I2", "I2_b"])
ax3.set_title('I2 toggle')

ax4 = plt.subplot(244)
ax4.plot(T, I3_a)
ax4.plot(T, I3_b)
ax4.legend(["I3_a = I3", "I3_b"])
ax4.set_title('I3 toggle')

ax5 = plt.subplot(212)
ax5.plot(T,out)
ax5.set_title('out')

plt.suptitle(f"S = [{S[1]},{S[0]}]")
plt.show()
"""


# plot

ax1 = plt.subplot(341)
ax1.plot(T, I0_a, color="#800000ff", alpha=0.75)
ax1.plot(T, I0_b, color="#999999ff", alpha=0.75)
ax1.legend(["$I_0$", "$\\overline{I_0}$"])
#ax1.set_title('$I_0$ toggle')
ax1.set_xlabel("Time [min]")
ax1.set_ylabel("Molecules")


ax2 = plt.subplot(342)
ax2.plot(T, I1_a, color = "#00ff00ff", alpha=0.75)
ax2.plot(T, I1_b, color = "#666666ff")#, alpha=0.75)
ax2.legend(["$I_1$", "$\\overline{I_1}$"])
#ax2.set_title('$I_1$ toggle')
ax2.set_xlabel("Time [min]")
ax2.set_ylabel("Molecules")


ax3 = plt.subplot(343)
ax3.plot(T, I2_a, color = "#0000ffff", alpha=0.75)
ax3.plot(T, I2_b, color = "#ecececfe")#, alpha=0.75)
ax3.legend(["$I_2$", "$\\overline{I_2}$"])
#ax3.set_title('$I_2$ toggle')
ax3.set_xlabel("Time [min]")
ax3.set_ylabel("Molecules")


ax4 = plt.subplot(344)
ax4.plot(T, I3_a, color = "#800080ff", alpha=0.75)
ax4.plot(T, I3_b, color = "#999999fc")#, alpha=0.75)
ax4.legend(["$I_3$", "$\\overline{I_3}$"])
#ax4.set_title('$I_3$ toggle')
ax4.set_xlabel("Time [min]")
ax4.set_ylabel("Molecules")


ax5 = plt.subplot(312)
ax5.plot(T,S0, color = "#ff6600ff", alpha=0.75)
ax5.plot(T,S1, color = "#ffff00ff")#, alpha=0.75)
ax5.legend(["$S_0$", "$S_1$"])
#ax5.set_title('Select inputs')
ax5.set_xlabel("Time [min]")
ax5.set_ylabel("Molecules")


ax6 = plt.subplot(313)
ax6.plot(T,out, color = "#8080805a", alpha=0.75)
#ax6.set_title('out')
ax6.legend(['out'])
ax6.set_xlabel("Time [min]")
ax6.set_ylabel("Molecules")

#step = int(self.N)
#ax6.plot(T[step::step], out[step::step], 'x')


#plt.suptitle("$out = \\overline{S}_1 \\overline{S}_0 I_0 \\vee \\overline{S}_1 S_0 I_1 \\vee S_1 \\overline{S}_0 I_2 \\vee S_1 S_0 I_3$")
plt.gcf().set_size_inches(15,10)
#plt.savefig(os.path.join("figs", "CBLB_ssa.pdf"), bbox_inches = 'tight')
plt.show()  