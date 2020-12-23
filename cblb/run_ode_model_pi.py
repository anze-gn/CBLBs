import os

from scipy.integrate import ode
import matplotlib.pyplot as plt

from models import *
from parameters import *

# [([S0, S1], [I0, I1, I2, I3]), ...]
states = [([0,0], [0,0,0,0]), 
          ([0,0], [1,0,0,0]), 
          ([1,0], [1,0,0,0]),
          ([1,0], [0,1,0,0]), 
          ([0,1], [0,1,0,0]),
          ([0,1], [0,0,1,0]), 
          ([1,1], [0,0,1,0]), 
          ([1,1], [0,0,0,1])]

# simulation parameters (for a single state)
t_end = 500
N = t_end
rho_x = 0
rho_y = 0

Y0 = np.zeros(49)

# number of cells: toggle switches
N_S0 = np.array([1,1])
N_S1 = np.array([1,1])

Y0[4:6] = N_S0
Y0[10:12] = N_S1

# number of cells: mux
#Y0[22-4+24:38-4+24] = 1 # number of cells ???
Y0[32:48] = 1 # number of cells


"""
simulations
"""

for iteration, state in enumerate(states):
    
    S = state[0]
    I = state[1]
    S0, S1 = S

    if iteration > 0 and states[iteration-1][0] == I:
        rho_S0_a, rho_S0_b, rho_S1_a, rho_S1_b = 0, 0, 0, 0
    else:
        rho_S0_a, rho_S0_b, rho_S1_a, rho_S1_b = (1-S0) * 5, S0*5, (1-S1)*5, S1*5

    rho_x, rho_y = 0,0
    params = (delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, delta_y, rho_x, rho_y, gamma_x, theta_x, r_X, r_Y, 
         rho_S0_a, rho_S0_b, rho_S1_a, rho_S1_b)

    if iteration:
        Y0 = Y_last[-1,:]
    
    Y0[12:16] = I

    # initialization
    T = np.linspace(0, t_end, N)
    t1 = t_end
    dt = t_end/N
    T = np.arange(0,t1+dt,dt)
    Y = np.zeros([1+N,49])
    Y[0,:] = Y0


    # simulation
    r = ode(PI_model_ODE).set_integrator('zvode', method='bdf')
    r.set_initial_value(Y0, T[0]).set_f_params(params)

    i = 1
    while r.successful() and r.t < t1:
        Y[i,:] = r.integrate(r.t+dt)
        i += 1

        # hold the state after half of the simulation time!
        if r.t > t1/2:
            params = (delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, delta_y, rho_x, rho_y, gamma_x, theta_x, r_X, r_Y, 
            0, 0, 0, 0)
            r.set_f_params(params)

    Y_last = Y
    if not iteration:
        Y_full = Y
        T_full = T
    else:
        Y_full = np.append(Y_full, Y, axis = 0)
        T_full = np.append(T_full, T + iteration * t_end, axis = 0)

Y = Y_full
T = T_full

I0, I1, I2, I3 = Y[:,12], Y[:,13], Y[:,14], Y[:,15]

S0_a, S0_b = Y[:,2], Y[:,3]
S1_a, S1_b = Y[:,8], Y[:,9]

out = Y[:,-1]

# plot

ax1 = plt.subplot(711)
ax1.plot(T, S0_a, color="#f00000ff", alpha=0.75)
ax1.plot(T, S0_b, color="#888888ff", alpha=0.75)
ax1.legend(["$S_0$", "$\\overline{S_0}$"])
#ax1.set_title('$S_0$ toggle')
ax1.set_xlabel("Time [min]")
ax1.set_ylabel("Concentrations [nM]")

ax2 = plt.subplot(712)
ax2.plot(T, S1_a, color = "#00f000ff", alpha=0.75)
ax2.plot(T, S1_b, color = "#888888ff", alpha=0.75)
ax2.legend(["$S_1$", "$\\overline{S_1}$"])
#ax2.set_title('$S_1$ toggle')
ax2.set_xlabel("Time [min]")
ax2.set_ylabel("Concentrations [nM]")

ax3 = plt.subplot(713)
ax3.plot(T, I0, color = "#000000ff", alpha=0.75)
ax3.legend(["$I_0$"])
#ax3.set_title('Data input I0')
ax3.set_xlabel("Time [min]")
ax3.set_ylabel("Concentrations [nM]")

ax4 = plt.subplot(714)
ax4.plot(T, I1, color = "#0000ffff", alpha=0.75)
ax4.legend(["$I_1$"])
#ax4.set_title('Data input I1')
ax4.set_xlabel("Time [min]")
ax4.set_ylabel("Concentrations [nM]")

ax5 = plt.subplot(715)
ax5.plot(T, I2, color = "#00ff00ff", alpha=0.75)
ax5.legend(["$I_2$"])
#ax5.set_title('Data input I2')
ax5.set_xlabel("Time [min]")
ax5.set_ylabel("Concentrations [nM]")

ax6 = plt.subplot(716)
ax6.plot(T, I3, color = "#ff0000ff", alpha=0.75)
ax6.legend(["$I_3$"])
#ax6.set_title('Data input I3')
ax6.set_xlabel("Time [min]")
ax6.set_ylabel("Concentrations [nM]")

ax7 = plt.subplot(717)
ax7.plot(T, out, color ="#8080805a", alpha=0.75)
#ax6.set_title('out')
ax7.legend(['out'])
ax7.set_xlabel("Time [min]")
ax7.set_ylabel("Concentrations [nM]")

#plt.suptitle("$out = \\overline{S}_1 \\overline{S}_0 I_0 \\vee \\overline{S}_1 S_0 I_1 \\vee S_1 \\overline{S}_0 I_2 \\vee S_1 S_0 I_3$")
plt.gcf().set_size_inches(15,15)
plt.savefig(os.path.join("figs", "PI_ode.pdf"), bbox_inches = 'tight')

plt.show()  
