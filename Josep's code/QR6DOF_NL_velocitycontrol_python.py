import setup_path 
import airsim
import numpy as np
import time
import matplotlib.pyplot as plt
import scipy.io as sio

# Function definitions
" Conversion Quaternion to Euler angles " 
def quaternion_to_euler(x, y, z, w):

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        X = np.arctan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Y = np.arcsin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Z = np.arctan2(t3, t4)

        return X, Y, Z

" Conversion Speed in Global Frame to in Body frame " 
def speed_global_to_body(x, y, z, phi, theta, psi):

        u = x*np.cos(theta)*np.cos(psi)+y*np.cos(theta)*np.sin(psi)-z*np.sin(theta)
        v = x*(np.cos(psi)*np.sin(theta)*np.sin(phi)-np.cos(phi)*np.sin(psi))+y*(np.sin(psi)*np.sin(theta)*np.sin(phi)+np.cos(phi)*np.cos(psi))+z*np.cos(theta)*np.sin(phi)
        w = x*(np.cos(phi)*np.cos(psi)*np.sin(theta)+np.sin(phi)*np.sin(psi))+y*(np.cos(phi)*np.sin(psi)*np.sin(theta)-np.sin(phi)*np.cos(psi))+z*np.cos(theta)*np.cos(phi)

        return u, v, w

" Get States and Time of QuadRotor "
def get_states_qr():
        
        s = client.getMultirotorState()
        [phi,theta,psi] = quaternion_to_euler(s.kinematics_estimated.orientation.x_val,\
                                          s.kinematics_estimated.orientation.y_val,\
                                          s.kinematics_estimated.orientation.z_val,\
                                          s.kinematics_estimated.orientation.w_val)
        [u,v,w] = speed_global_to_body(s.kinematics_estimated.linear_velocity.x_val,\
                                       -s.kinematics_estimated.linear_velocity.y_val,\
                                       -s.kinematics_estimated.linear_velocity.z_val, phi, -theta, -psi)
        x = np.array([s.kinematics_estimated.position.x_val,-s.kinematics_estimated.position.y_val,\
                      -s.kinematics_estimated.position.z_val,phi,\
                      -theta,-psi,\
                      u,v,\
                      w,s.kinematics_estimated.angular_velocity.x_val,\
                      -s.kinematics_estimated.angular_velocity.y_val,-s.kinematics_estimated.angular_velocity.z_val])
        t = s.timestamp/1e9

        return x,t

# Connect to AirSim, Confirm Connection, Enable API Control, Arm QuadRotor  
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Control time step, simulation time, physical Airsim clock speed
T = 0.001
Time = 10
len = Time/T

# Create empty array to save simulation data
x = np.empty((int(len)+1,12))
t = np.empty(int(len)+1)
u = np.empty((int(len)+1,4))

# Load feeback, integrator, ... matrices
mat_contents = sio.loadmat('mat_controlvelocidad_params.mat')
C1           = mat_contents['C1']
C2           = mat_contents['C2']
K1           = mat_contents['K1']
K2           = mat_contents['K2']
Ki1          = mat_contents['Ki1']
Ki2          = mat_contents['Ki2']
u0           = np.transpose(mat_contents['u0'])
mixer        = mat_contents['Mixer']
x_ctr_2_1    = mat_contents['x_ctr_2_1']
x_ctr_2_2    = mat_contents['x_ctr_2_2']

# Control Loop
" Reference "
R_u   =  5; t_u   = 2
R_v   =  5; t_v   = 2
R_Z   =  5; t_Z   = 0
R_psi =  -30*np.pi/180; t_psi = 2
Ref2                  = np.zeros((int(len),2))
Ref1                  = np.zeros((int(len),4))
Ref2[int(t_u/T):,0]   = R_u
Ref2[int(t_v/T):,1]   = R_v
Ref1[int(t_Z/T):,0]   = R_Z
Ref1[int(t_psi/T):,3] = R_psi
" Integrator "
xi1 = np.zeros(Ref1.shape[1])
xi2 = np.zeros(Ref2.shape[1])
" Loop "
for i in range(int(len)):
    [x[i],t[i]] = get_states_qr()
    'External Loop'
    dxi           = Ref2[i]-C2@x_ctr_2_2@x[i]  
    xi2           = xi2+dxi*T
    Ref1[i,(1,2)] = -K2@x_ctr_2_2@x[i]+Ki2@xi2
    'Internal Loop'
    dxi  = Ref1[i]-C1@x_ctr_2_1@x[i]  
    xi1   = xi1+dxi*T
    u[i] = -K1@x_ctr_2_1@x[i]+Ki1@xi1+u0
    
    pwm = np.clip(mixer@u[i]/(669.8574**2),0,1)  
    client.moveByMotorPWMsAsync(pwm[0], pwm[1], pwm[2], pwm[3], 5)

    time.sleep((i+1)*T-(t[i]-t[0]))

# Final states, control actions and times
[x[i+1],t[i+1]] = get_states_qr()
u[i+1] = u[i]
t = (t-t[0])

# Save simulation data
sio.savemat('mat_controlvelocidad_python_sim.mat', {'x':x,'t':t,'u':u})

plt.rcParams['legend.fontsize'] = 10

"""fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(x, y, z, label='parametric curve')
ax.legend()
plt.show()"""

plt.plot(t,x[:,0])
plt.plot(t,x[:,1])
plt.plot(t,x[:,2])
plt.grid(True)
plt.show()
