%% SYSTEM PARAMETERS

% Geometric Parameters
RADS2RPM = 60/(2*pi);
RAD2DEG = 180/pi;

n_motor = 4; % Number of motors
l = 0.2275; % Arm length [m]
l_arm_x = l*sqrt(2)/2; % Arm length in X axis [m]
l_arm_y = l*sqrt(2)/2; % Arm length in Y axis [m]
l_arm_z = 0.025; % Arm length in Z axis [m]
l_box_x = 0.18; % Central body length in X axis [m]
l_box_y = 0.11; % Central body length in Y axis [m]
l_box_z = 0.04; % Central body length in Z axis [m]
l_feet = 0.1393; % Feet length in Z axis [m]
Diam = 0.2286; % Rotor diameter [m]
Rad = Diam/2; % Rotor radius [m]

% Mass Parameters
m = 1; % Mass [kg]
m_motor = 0.055; % Motor mass [kg]
m_box = m-n_motor*m_motor; % Central body mass [kg]
g = 9.8; % Gravity [m/s^2]
Ixx = m_box/12*(l_box_y^2+l_box_z^2)+(l_arm_y^2+l_arm_z^2)*m_motor*n_motor;
% Inertia in X axis [kg m^2]
Iyy = m_box/12*(l_box_x^2+l_box_z^2)+(l_arm_x^2+l_arm_z^2)*m_motor*n_motor;
% Inertia in Y axis [kg m^2]
Izz = m_box/12*(l_box_x^2+l_box_y^2)+(l_arm_x^2+l_arm_y^2)*m_motor*n_motor;
% Inertia in Z axis [kg m^2]
Ir = 2.03e-5; % Rotor inertia around spinning axis [kg m^2]

% Motor Parameters
max_rpm = 6396.667; % Rotor max RPM
max_omega = max_rpm/RADS2RPM; % Rotor max angular velocity [rad/s]
Tm = 0.005; % Motor low pass filter

% Aerodynamics Parameters
CT = 0.109919; % Traction coefficient [-]
CP = 0.040164; % Moment coefficient [-]
rho = 1.225; % Air density [kg/m^3]
k1 = CT*rho*Diam^4;
b1 = CP*rho*Diam^5/(2*pi);
Tmax = k1*(max_rpm/60)^2; % Max traction [N]
Qmax = b1*(max_rpm/60)^2; % Max moment [Nm]
k = Tmax/(max_omega^2); % Traction coefficient
b = Qmax/(max_omega^2); % Moment coefficient
c = (0.04-0.0035); % Lumped Drag constant
KB = 2; % Fountain effect (:2)

% Contact Parameters
max_disp_a = 0.001; % Max displacement in contact [m]
n_a = 4; % Number of contacts [-]
xi = 0.95; % Relative damping [-]
ka = m*g/(n_a*max_disp_a); % Contact stiffness [N/m]
ca = 2*m*sqrt(ka/m)*xi*1/sqrt(n_a); % Contact damping [Ns/m]
mua = 0.5; % Coulomb friction coefficient [-]

% Mixer ( [F,taux,tay,tauz]->[omega1,omega2,omega3,omega4] )
% invMixer ( [omega1,omega2,omega3,omega4]->[F,taux,tay,tauz] )
Mixer = [ 1/(4*k) -(1/(2*sqrt(2)*l*k)) -(1/(2*sqrt(2)*l*k)) -(1/(4*b))
1/(4*k) (1/(2*sqrt(2)*l*k)) (1/(2*sqrt(2)*l*k)) -(1/(4*b))
1/(4*k) (1/(2*sqrt(2)*l*k)) -(1/(2*sqrt(2)*l*k)) (1/(4*b))
1/(4*k) -(1/(2*sqrt(2)*l*k)) (1/(2*sqrt(2)*l*k)) (1/(4*b)) ];

invMixer = [ k k k k
-sqrt(2)*l*k/2 sqrt(2)*l*k/2 sqrt(2)*l*k/2 -sqrt(2)*l*k/2
-sqrt(2)*l*k/2 sqrt(2)*l*k/2 -sqrt(2)*l*k/2 sqrt(2)*l*k/2
-b -b b ...
b ];


RADS2RPM = 60/(2*pi);
RAD2DEG = 180/pi;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% QUADROTOR POSITION CONTROL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% SYSTEM MODEL
%% SIMULATION PARAMETERS
% Simulation time
Time = 12;
% Acquisition time, discrete Kalman algorithm time interval
T = 0.0003;
% Time vector
t = (0:T:Time)';
% Controlled variables references
R_X = 5; t_X = 8;
R_Y = 5; t_Y = 4;
R_Z = 5; t_Z = 0;
R_psi = pi; t_psi = 3;
% Disturbances
R_uwg = 0; t_uwg = 0;
R_vwg = 0; t_vwg = 0;
R_wwg = 0; t_wwg = 0;
R_Ax = 0; t_Ax = 0;
R_Ay = 0; t_Ay = 0;
R_Az = 0; t_Az = 0;
R_tauwx = 0; t_tauwx = 0;
R_tauwy = 0; t_tauwy = 0;
R_tauwz = 0; t_tauwz = 0;
% Initial conditions
x0 = [0;0;l_feet;0;0;0;0;0;0;0;0;0];

%% STATE SPACE SYSTEM
% States x = [x,y,z,phi,theta,psi,u,v,w,p,q,r]
% States in hover equilibrium x = [0 0 0 0 0 0 0 0 0 0 0 0]
% Control actions u = [F,taux,tauy,tauz]
% Control actions in hover equilibrium
F_0 = m*g; tauX_0 = 0; tauY_0 = 0; tauZ_0 = 0;
u0 = [F_0; tauX_0; tauY_0; tauZ_0];

A = [ 0 0 0 0 0 0 1 0 0 0 0 0
0 0 0 0 0 0 0 1 0 0 0 0
0 0 0 0 0 0 0 0 1 0 0 0
0 0 0 0 0 0 0 0 0 1 0 0
0 0 0 0 0 0 0 0 0 0 1 0
0 0 0 0 0 0 0 0 0 0 0 1
0 0 0 0 g 0 0 0 0 0 0 0
0 0 0 -g 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 0 0 0 ];

B = [ 0 0 0 0
0 0 0 0
0 0 0 0
0 0 0 0
0 0 0 0
0 0 0 0
0 0 0 0
0 0 0 0
1/m 0 0 0
0 1/Ixx 0 0
0 0 1/Iyy 0
0 0 0 1/Izz ];

C = diag(ones(size(A,1),1));

D = zeros(size(C,1),size(B,2));

% To access all states from state space block in Simulink
Cs = eye(size(C,2));
Ds = zeros(size(A,2),size(B,2));
%% LQR CONTROLLER DESIGN

% INTERNAL CONTROLLER
% Considered states in internal controller
x_ctr = [3,4,5,6,9,10,11,12]';
x_ctr_2_1 = zeros(size(x_ctr,1),size(A,1));
x_ctr_2_1(sub2ind(size(x_ctr_2_1),(1:size(x_ctr_2_1,1))',x_ctr)) = 1;
% Control Actions in internal controller
u_ctr = (1:4)';
n_per = 9; % Number of disturbances
u_ctr_2 = zeros(n_per+size(B,2),n_per+size(u_ctr,1));
u_ctr_2(sub2ind(size(u_ctr_2),(1:n_per)',(1:n_per)')) = 1;
u_ctr_2(sub2ind(size(u_ctr_2),n_per+u_ctr,n_per+...
(1:size(u_ctr_2,2)-n_per)')) = 1;
u_ctr_2 = u_ctr_2';
% Outputs in internal controller
y_ctr = [3,4,5,6]';
% State space system internal controller
A1 = A(x_ctr,x_ctr);
B1 = B(x_ctr,u_ctr);
C1 = C(y_ctr,x_ctr);
D1 = D(y_ctr,u_ctr);
 % Extended system
Ab1 = [A1 zeros(size(A1,1),size(C1,1)); -C1 zeros(size(C1,1),size(C1,1))];
Bb1 = [B1; zeros(size(C1,1),size(B1,2))];
% Extended system controllability
Co = ctrb(Ab1,Bb1);
unco = length(Ab1) - rank(Co);
fprintf('Uncontrollable states internal extended system: %g\n',unco)
% Q y R LQR Matrices
Q = blkdiag(0,1,1,0,0,0,0,0,0.6,60,60,1e-5);
R = diag([1,1,1,1])*0.01;
% Internal LQR
[Kb,tst1,tst2] = lqr(Ab1,Bb1,Q,R);
K1 = Kb(:,1:(size(A1,2)));
Ki1 = -Kb(:,end-size(C1,1)+1:end);
fprintf('Extended internal system poles:\n')
eig(Ab1-Bb1*Kb)

% EXTERNAL REGULATOR
% States considered in external regulator
x_ctr = [1,2,7,8]';
x_ctr_2_2 = zeros(size(x_ctr,1),size(A,1));
x_ctr_2_2(sub2ind(size(x_ctr_2_2),(1:size(x_ctr_2_2,1))',x_ctr)) = 1;
% Control actions in external regulator
u_ctr = [4,5]';
% LQR external outputs
y_ctr = [1,2]';
% State space system external controller
A2 = A(x_ctr,x_ctr);
B2 = A(x_ctr,u_ctr);
C2 = C(y_ctr,x_ctr);
D2 = zeros(size(y_ctr,1),size(u_ctr,1));
% Extended external system
Ab2 = [A2 zeros(size(A2,1),size(C2,1)); -C2 zeros(size(C2,1),size(C2,1))];
Bb2 = [B2; zeros(size(C2,1),size(B2,2))];
% Extended system controllability
Co = ctrb(Ab2,Bb2);
unco = length(Ab2) - rank(Co);
fprintf('Uncontrollable states internal extended system: %g\n',unco)
% Q y R LQR Matrices
Q = blkdiag(0,0,0,0,0.07,0.07);
R = diag([1,1]);
% External LQR
[Kb,tst1,tst2] = lqr(Ab2,Bb2,Q,R);
K2b = Kb(:,1:(size(A2,2))); K2xy = K2b(2,1); K2uv = K2b(2,3);
Ki2b = -Kb(:,end-size(C2,1)+1:end); Ki2 = Ki2b(2,1);
fprintf('Extended external system poles:\n')
eig(Ab2-Bb2*Kb)
save('mat_controlposicion_params.mat','K1','K2b','Ki1','Ki2b','C1','C2',...
'u0','Mixer','x_ctr_2_1','x_ctr_2_2','max_omega','x0',...
'Izz','Iyy','Ixx','m','g','invMixer','K2xy','K2uv',...
'Ki2')