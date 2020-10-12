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

" Non lineal dynamic model "
def dyn_model(xk,uk,T,Ixx,Iyy,Izz,m,g):

        phi = xk[3]; theta = xk[4];  psi = xk[5]
        u   = xk[6]; v     = xk[7];  w   = xk[8]
        p   = xk[9]; q     = xk[10]; r   = xk[11]

        Tr = uk[0]; MxT = uk[1]; MyT = uk[2]; MzT = uk[3]
  
        xDot = (u*np.cos(theta)*np.cos(psi)+v*(np.cos(psi)*np.sin(theta)*\
                    np.sin(phi)+(-1)*np.cos(phi)*np.sin(psi))+w*(np.cos(phi)*np.cos(psi)*\
                    np.sin(theta)+np.sin(phi)*np.sin(psi)))
        yDot = (u*np.cos(theta)*np.sin(psi)+w*((-1)*np.cos(psi)*np.sin(phi)+\
                    np.cos(phi)*np.sin(theta)*np.sin(psi))+v*(np.cos(phi)*np.cos(psi)+np.sin(theta)*\
                    np.sin(phi)*np.sin(psi)))
        zDot = (w*np.cos(theta)*np.cos(phi)+(-1)*u*np.sin(theta)+v*np.cos(theta)*\
                    np.sin(phi))

        phiDot   = (p+r*np.cos(phi)*np.tan(theta)+q*np.sin(phi)*np.tan(theta))
        thetaDot = (q*np.cos(phi)+(-1)*r*np.sin(phi))
        psiDot   = (r*np.cos(phi)/np.cos(theta)+q/np.cos(theta)*np.sin(phi))

        uDot = (g*np.sin(theta)+r*v+(-1)*q*w)
        vDot = (-g*np.cos(theta)*np.sin(phi)+(-1)*r*u+p*w)
        wDot = (-g*np.cos(theta)*np.cos(phi)+Tr*m**(-1)+q*u+(-1)*p*v)

        pDot = (Ixx**(-1)*(Iyy*q*r+(-1)*Izz*q*r+MxT))
        qDot = (Iyy**(-1)*((-1)*Ixx*p*r+Izz*p*r+MyT))
        rDot = (Izz**(-1)*(Ixx*p*q+(-1)*Iyy*p*q+MzT))

        xkmDot = np.array([xDot,yDot,zDot,phiDot,thetaDot,psiDot,uDot,vDot,wDot,pDot,qDot,rDot])
        xkm    = xk+xkmDot*T
        
        return xkm

" Linearization around estimate "
def lin_model(xk,Ixx,Iyy,Izz,m,g):

        phi = xk[3]; theta = xk[4];  psi = xk[5]
        u   = xk[6]; v     = xk[7];  w   = xk[8]
        p   = xk[9]; q     = xk[10]; r   = xk[11]
        
        Ak = np.array([[0,0,0,\
            w*((-1)*np.cos(psi)*np.sin(theta)*np.sin(phi)+np.cos(phi)*np.sin(psi))+\
            v*(np.cos(phi)*np.cos(psi)*np.sin(theta)+np.sin(phi)*np.sin(psi)),\
            w*np.cos(theta)*np.cos(phi)*np.cos(psi)+(-1)*u*np.cos(psi)*np.sin(theta)+\
            v*np.cos(theta)*np.cos(psi)*np.sin(phi),\
            (-1)*u*np.cos(theta)*np.sin(psi)+\
            w*(np.cos(psi)*np.sin(phi)+(-1)*np.cos(phi)*np.sin(theta)*np.sin(psi))+\
            v*((-1)*np.cos(phi)*np.cos(psi)+(-1)*np.sin(theta)*np.sin(phi)*np.sin(psi)),\
            np.cos(theta)*np.cos(psi),\
            np.cos(psi)*np.sin(theta)*np.sin(phi)+(-1)*np.cos(phi)*np.sin(psi),\
            np.cos(phi)*np.cos(psi)*np.sin(theta)+np.sin(phi)*np.sin(psi),0,0,0],\
        [0,0,0,\
            v*((-1)*np.cos(psi)*np.sin(phi)+np.cos(phi)*np.sin(theta)*np.sin(psi))+\
            w*((-1)*np.cos(phi)*np.cos(psi)+(-1)*np.sin(theta)*np.sin(phi)*np.sin(psi)),\
            w*np.cos(theta)*np.cos(phi)*np.sin(psi)+(-1)*u*np.sin(theta)*np.sin(psi)+\
            v*np.cos(theta)*np.sin(phi)*np.sin(psi),\
            u*np.cos(theta)*np.cos(psi)+\
            v*(np.cos(psi)*np.sin(theta)*np.sin(phi)+(-1)*np.cos(phi)*np.sin(psi))+\
            w*(np.cos(phi)*np.cos(psi)*np.sin(theta)+np.sin(phi)*np.sin(psi)),\
            np.cos(theta)*np.sin(psi),\
            np.cos(phi)*np.cos(psi)+np.sin(theta)*np.sin(phi)*np.sin(psi),\
            (-1)*np.cos(psi)*np.sin(phi)+np.cos(phi)*np.sin(theta)*np.sin(psi),0,0,0],\
        [0,0,0,\
            v*np.cos(theta)*np.cos(phi)+(-1)*w*np.cos(theta)*np.sin(phi),\
            (-1)*u*np.cos(theta)+(-1)*w*np.cos(phi)*np.sin(theta)+(-1)* \
            v*np.sin(theta)*np.sin(phi),\
            0,(-1)*np.sin(theta),np.cos(theta)*np.sin(phi),np.cos(theta)*np.cos(phi),0,0,0],\
        [0,0,0,\
            q*np.cos(phi)*np.tan(theta)+(-1)*r*np.sin(phi)*np.tan(theta),\
            r*np.cos(phi)/np.cos(theta)**2+q/np.cos(theta)**2*np.sin(phi),\
            0,0,0,0,1,np.sin(phi)*np.tan(theta),np.cos(phi)*np.tan(theta)],\
        [0,0,0,\
            (-1)*r*np.cos(phi)+(-1)*q*np.sin(phi),0,0,0,0,0,0,\
            np.cos(phi),(-1)*np.sin(phi)],\
        [0,0,0,\
            q*np.cos(phi)/np.cos(theta)+(-1)*r/np.cos(theta)*np.sin(phi),\
            r*np.cos(phi)/np.cos(theta)*np.tan(theta)+\
            q/np.cos(theta)*np.sin(phi)*np.tan(theta),0,0,0,0,0,np.cos(theta)**(-1)*np.sin(phi),\
            np.cos(phi)/np.cos(theta)],\
        [0,0,0,\
            0,g*np.cos(theta),0,0,r,(-1)*q,0,(-1)*w,v],\
        [0,0,0,\
            (-1)*g*np.cos(theta)*np.cos(phi),g*np.sin(theta)*np.sin(phi),0,\
            (-1)*r,0,p,w,0,(-1)*u],\
        [0,0,0,\
            g*np.cos(theta)*np.sin(phi),g*np.cos(phi)*np.sin(theta),0,\
            q,(-1)*p,0,(-1)*v,u,0],\
        [0,0,0,\
            0,0,0,0,0,0,0,\
            Ixx**(-1)*(Iyy*r+(-1)*Izz*r),Ixx**(-1)*(Iyy*q+(-1)*Izz*q)],\
        [0,0,0,\
            0,0,0,0,0, \
            0,Iyy**(-1)*((-1)*Ixx*r+Izz*r),0,Iyy**(-1)*((-1)*Ixx*p+ \
            Izz*p)],\
        [0,0,0,\
            0,0,0,0,0,0,Izz**(-1)*(Ixx*q+(-1) \
            *Iyy*q),Izz**(-1)*(Ixx*p+(-1)*Iyy*p),0]])
        
        return Ak

" Yaw state and reference conditioning "
def yaw_cond(psi,psi_prev):

        if np.sign(psi)!=np.sign(psi_prev):
                if (np.absolute(psi)>np.pi/4)and(np.absolute(psi_prev)>np.pi/4):
                        if psi<psi_prev:
                                add = 2*np.pi
                        else:
                                add = -2*np.pi
                else:
                        add = 0
        else:
                add = 0
        
        return add

# Connect to AirSim, Confirm Connection, Enable API Control, Arm QuadRotor  
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

# Control time step, simulation time, physical Airsim clock speed
T = 0.0003
Time = 10
len = Time/T

# Create empty array to save simulation data
x   = np.empty((int(len)+1,12))
t   = np.empty(int(len)+1)
u   = np.empty((int(len)+1,4))
yk  = np.empty((int(len)+1,6))
xkm = np.empty((int(len)+1,12))
xk  = np.empty((int(len)+1,12))

# Load feeback, integrator, ... matrices
mat_contents = sio.loadmat('mat_controlvelocidad_params.mat')
C1           = mat_contents['C1']
C2           = mat_contents['C2']
Ck           = mat_contents['Ck']
K1           = mat_contents['K1']
K2           = mat_contents['K2']
Ki1          = mat_contents['Ki1']
Ki2          = mat_contents['Ki2']
u0           = np.transpose(mat_contents['u0']).flatten('F')
x0           = np.transpose(mat_contents['x0'])
mixer        = mat_contents['Mixer']
invmixer     = mat_contents['invMixer']
x_ctr_2_1    = mat_contents['x_ctr_2_1']
x_ctr_2_2    = mat_contents['x_ctr_2_2']
Pk0          = mat_contents['Pk0']
Qk           = mat_contents['Qk']
Rk           = mat_contents['Rk']
max_omega    = float(mat_contents['max_omega'])
vk           = mat_contents['vk']
Izz          = float(mat_contents['Izz'])
Iyy          = float(mat_contents['Iyy'])
Ixx          = float(mat_contents['Ixx'])
m            = float(mat_contents['m'])
g            = float(mat_contents['g'])

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
"Kalman initial estimates"
xk[0] = x0
uk0   = u0
Pk    = Pk0
"Yaw reference conditioning"
adder = 0
psi_prev = x0[0,5]
" Loop "
for i in range(int(len)):
    [x[i],t[i]] = get_states_qr()
    'Check sign of yaw angle'
    adder    += yaw_cond(x[i,5],psi_prev)
    psi_prev = x[i,5]
    x[i,5]   = x[i,5]+adder
    'Extended Kalman Filter'
    yk[i]   = Ck@x[i]+vk[i,1:]
    xkm[i]  = np.transpose(dyn_model(xk[i],uk0,T,Ixx,Iyy,Izz,m,g))
    Ak      = lin_model(xk[i],Ixx,Iyy,Izz,m,g)
    PkmDot  = Ak@Pk+Pk@np.transpose(Ak)+Qk
    Pkm     = Pk+PkmDot*T
    Kk      = Pkm@np.transpose(Ck)@np.linalg.inv((Ck@Pkm@np.transpose(Ck)+Rk))
    xk[i+1] = xkm[i]+Kk@(yk[i]-Ck@xkm[i])
    Pk      = (np.identity(12)-Kk@Ck)@Pkm;
    'External Loop'
    dxi           = Ref2[i]-C2@x_ctr_2_2@x[i]  
    xi2           = xi2+dxi*T
    Ref1[i,(1,2)] = -K2@x_ctr_2_2@x[i]+Ki2@xi2
    'Yaw reference between range -180+psi, 180+psi'
    Ref1[i,3] = np.round((xk[i,5]-Ref1[i,3])/(2*np.pi))*2*np.pi+Ref1[i,3]
    'Internal Loop'
    dxi  = Ref1[i]-C1@x_ctr_2_1@x[i]  
    xi1   = xi1+dxi*T
    u[i] = -K1@x_ctr_2_1@x[i]+Ki1@xi1+u0
    'PWM inputs'
    pwm  = np.clip(mixer@u[i]/(max_omega**2),0,1)
    u[i] = invmixer@np.clip(mixer@u[i],0,max_omega**2)
    uk0  = u[i].flatten('F')
    'Control API'
    client.moveByMotorPWMsAsync(pwm[0], pwm[1], pwm[2], pwm[3], 5)
    'Sellep the control algorithm until next iterations'
    time.sleep((i+1)*T-(t[i]-t[0]))

# Final states, control actions and times
[x[i+1],t[i+1]] = get_states_qr()
u[i+1]          = u[i]
yk[i+1]         = yk[i]
xkm[i+1]        = xkm[i]
t               = (t-t[0])

# Save simulation data
sio.savemat('mat_controlvelocidad_python_sim.mat', {'x':x,'t':t,'u':u,'xk':xk,'xkm':xkm,'yk':yk})

plt.plot(t,yk[:,0],color='red')
plt.plot(t,yk[:,1],color='green')
plt.plot(t,yk[:,2],color='blue')
plt.plot(t,xk[:,0],color='black')
plt.plot(t,xk[:,1],color='black')
plt.plot(t,xk[:,2],color='black')
plt.grid(True)
plt.show()

plt.plot(t,yk[:,3],color='red')
plt.plot(t,yk[:,4],color='green')
plt.plot(t,yk[:,5],color='blue')
plt.plot(t,xk[:,3],color='black')
plt.plot(t,xk[:,4],color='black')
plt.plot(t,xk[:,5],color='black')
plt.grid(True)
plt.show()
