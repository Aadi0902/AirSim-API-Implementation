%% SYSTEM PARAMETERS
2
3 % Geometric Parameters
4 n_motor = 4; % Number of motors
5 l = 0.2275; % Arm length [m]
6 l_arm_x = l*sqrt(2)/2; % Arm length in X axis [m]
7 l_arm_y = l*sqrt(2)/2; % Arm length in Y axis [m]
8 l_arm_z = 0.025; % Arm length in Z axis [m]
9 l_box_x = 0.18; % Central body length in X axis [m]
10 l_box_y = 0.11; % Central body length in Y axis [m]
11 l_box_z = 0.04; % Central body length in Z axis [m]
12 l_feet = 0.1393; % Feet length in Z axis [m]
13 Diam = 0.2286; % Rotor diameter [m]
14 Rad = Diam/2; % Rotor radius [m]
15
16 % Mass Parameters
17 m = 1; % Mass [kg]
18 m_motor = 0.055; % Motor mass [kg]
19 m_box = m-n_motor*m_motor; % Central body mass [kg]
20 g = 9.8; % Gravity [m/s^2]
21 Ixx = m_box/12*(l_box_y^2+l_box_z^2)+(l_arm_y^2+l_arm_z^2)*m_motor*n_motor;
22 % Inertia in X axis [kg m^2]
23 Iyy = m_box/12*(l_box_x^2+l_box_z^2)+(l_arm_x^2+l_arm_z^2)*m_motor*n_motor;
24 % Inertia in Y axis [kg m^2]
25 Izz = m_box/12*(l_box_x^2+l_box_y^2)+(l_arm_x^2+l_arm_y^2)*m_motor*n_motor;
26 % Inertia in Z axis [kg m^2]
27 Ir = 2.03e-5; % Rotor inertia around spinning axis [kg m^2]
28
29 % Motor Parameters
30 max_rpm = 6396.667; % Rotor max RPM
31 max_omega = max_rpm/RADS2RPM; % Rotor max angular velocity [rad/s]
32 Tm = 0.005; % Motor low pass filter
33
34 % Aerodynamics Parameters
35 CT = 0.109919; % Traction coefficient [-]
36 CP = 0.040164; % Moment coefficient [-]
37 rho = 1.225; % Air density [kg/m^3]
38 k1 = CT*rho*Diam^4;
39 b1 = CP*rho*Diam^5/(2*pi);
40 Tmax = k1*(max_rpm/60)^2; % Max traction [N]
41 Qmax = b1*(max_rpm/60)^2; % Max moment [Nm]
42 k = Tmax/(max_omega^2); % Traction coefficient
43 b = Qmax/(max_omega^2); % Moment coefficient
44 c = (0.04-0.0035); % Lumped Drag constant
45 KB = 2; % Fountain effect (:2)
46
47 % Contact Parameters
48 max_disp_a = 0.001; % Max displacement in contact [m]
49 n_a = 4; % Number of contacts [-]
50 xi = 0.95; % Relative damping [-]
51 ka = m*g/(n_a*max_disp_a); % Contact stiffness [N/m]
52 ca = 2*m*sqrt(ka/m)*xi*1/sqrt(n_a); % Contact damping [Ns/m]
53 mua = 0.5; % Coulomb friction coefficient [-]
54
55 % Mixer ( [F,taux,tay,tauz]->[omega1,omega2,omega3,omega4] )
56 % invMixer ( [omega1,omega2,omega3,omega4]->[F,taux,tay,tauz] )
57 Mixer = [ 1/(4*k) -(1/(2*sqrt(2)*l*k)) -(1/(2*sqrt(2)*l*k)) -(1/(4*b))
58 1/(4*k) (1/(2*sqrt(2)*l*k)) (1/(2*sqrt(2)*l*k)) -(1/(4*b))
59 1/(4*k) (1/(2*sqrt(2)*l*k)) -(1/(2*sqrt(2)*l*k)) (1/(4*b))
60 1/(4*k) -(1/(2*sqrt(2)*l*k)) (1/(2*sqrt(2)*l*k)) (1/(4*b)) ];
61
62 invMixer = [ k k k k
63 -sqrt(2)*l*k/2 sqrt(2)*l*k/2 sqrt(2)*l*k/2 -sqrt(2)*l*k/2
64 -sqrt(2)*l*k/2 sqrt(2)*l*k/2 -sqrt(2)*l*k/2 sqrt(2)*l*k/2
65 -b -b b ...
b ];


close all
clear

RADS2RPM = 60/(2*pi);
RAD2DEG = 180/pi;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% QUADROTOR POSITION CONTROL %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% SYSTEM MODEL
%% SIMULATION PARAMETERS
% Simulation time
16 Time = 12;
17 % Acquisition time, discrete Kalman algorithm time interval
18 T = 0.0003;
19 % Time vector
20 t = (0:T:Time)';
21 % Controlled variables references
22 R_X = 5; t_X = 8;
23 R_Y = 5; t_Y = 4;
24 R_Z = 5; t_Z = 0;
25 R_psi = pi; t_psi = 3;
26 % Disturbances
27 R_uwg = 0; t_uwg = 0;
28 R_vwg = 0; t_vwg = 0;
29 R_wwg = 0; t_wwg = 0;
30 R_Ax = 0; t_Ax = 0;
31 R_Ay = 0; t_Ay = 0;
32 R_Az = 0; t_Az = 0;
33 R_tauwx = 0; t_tauwx = 0;
34 R_tauwy = 0; t_tauwy = 0;
35 R_tauwz = 0; t_tauwz = 0;
36 % Initial conditions
37 x0 = [0;0;l_feet;0;0;0;0;0;0;0;0;0];
38
39 %% STATE SPACE SYSTEM
40 % States x = [x,y,z,phi,theta,psi,u,v,w,p,q,r]
41 % States in hover equilibrium x = [0 0 0 0 0 0 0 0 0 0 0 0]
42 % Control actions u = [F,taux,tauy,tauz]
43 % Control actions in hover equilibrium
44 F_0 = m*g; tauX_0 = 0; tauY_0 = 0; tauZ_0 = 0;
45 u0 = [F_0; tauX_0; tauY_0; tauZ_0];
46
47 A = [ 0 0 0 0 0 0 1 0 0 0 0 0
48 0 0 0 0 0 0 0 1 0 0 0 0
49 0 0 0 0 0 0 0 0 1 0 0 0
50 0 0 0 0 0 0 0 0 0 1 0 0
51 0 0 0 0 0 0 0 0 0 0 1 0
52 0 0 0 0 0 0 0 0 0 0 0 1
53 0 0 0 0 g 0 0 0 0 0 0 0
54 0 0 0 -g 0 0 0 0 0 0 0 0
55 0 0 0 0 0 0 0 0 0 0 0 0
56 0 0 0 0 0 0 0 0 0 0 0 0
57 0 0 0 0 0 0 0 0 0 0 0 0
58 0 0 0 0 0 0 0 0 0 0 0 0 ];
59
60 B = [ 0 0 0 0
61 0 0 0 0
62 0 0 0 0
63 0 0 0 0
64 0 0 0 0
65 0 0 0 0
66 0 0 0 0
67 0 0 0 0
68 1/m 0 0 0
69 0 1/Ixx 0 0
70 0 0 1/Iyy 0
71 0 0 0 1/Izz ];
72
73 C = diag(ones(size(A,1),1));
74
75 D = zeros(size(C,1),size(B,2));
76
77 % To access all states from state space block in Simulink
78 Cs = eye(size(C,2));
79 Ds = zeros(size(A,2),size(B,2));
81 %% LQR CONTROLLER DESIGN
82
83 % INTERNAL CONTROLLER
84 % Considered states in internal controller
85 x_ctr = [3,4,5,6,9,10,11,12]';
86 x_ctr_2_1 = zeros(size(x_ctr,1),size(A,1));
87 x_ctr_2_1(sub2ind(size(x_ctr_2_1),(1:size(x_ctr_2_1,1))',x_ctr)) = 1;
88 % Control Actions in internal controller
89 u_ctr = (1:4)';
90 n_per = 9; % Number of disturbances
91 u_ctr_2 = zeros(n_per+size(B,2),n_per+size(u_ctr,1));
92 u_ctr_2(sub2ind(size(u_ctr_2),(1:n_per)',(1:n_per)')) = 1;
93 u_ctr_2(sub2ind(size(u_ctr_2),n_per+u_ctr,n_per+...
94 (1:size(u_ctr_2,2)-n_per)')) = 1;
95 u_ctr_2 = u_ctr_2';
96 % Outputs in internal controller
97 y_ctr = [3,4,5,6]';
98 % State space system internal controller
99 A1 = A(x_ctr,x_ctr);
100 B1 = B(x_ctr,u_ctr);
101 C1 = C(y_ctr,x_ctr);
102 D1 = D(y_ctr,u_ctr);
103 % Extended system
104 Ab1 = [A1 zeros(size(A1,1),size(C1,1)); -C1 zeros(size(C1,1),size(C1,1))];
105 Bb1 = [B1; zeros(size(C1,1),size(B1,2))];
106 % Extended system controllability
107 Co = ctrb(Ab1,Bb1);
108 unco = length(Ab1) - rank(Co);
109 fprintf('Uncontrollable states internal extended system: %g\n',unco)
110 % Q y R LQR Matrices
111 Q = blkdiag(0,1,1,0,0,0,0,0,0.6,60,60,1e-5);
112 R = diag([1,1,1,1])*0.01;
113 % Internal LQR
114 [Kb,:,:] = lqr(Ab1,Bb1,Q,R);
115 K1 = Kb(:,1:(size(A1,2)));
116 Ki1 = -Kb(:,end-size(C1,1)+1:end);
117 fprintf('Extended internal system poles:\n')
118 eig(Ab1-Bb1*Kb)
119
120 % EXTERNAL REGULATOR
121 % States considered in external regulator
122 x_ctr = [1,2,7,8]';
123 x_ctr_2_2 = zeros(size(x_ctr,1),size(A,1));
124 x_ctr_2_2(sub2ind(size(x_ctr_2_2),(1:size(x_ctr_2_2,1))',x_ctr)) = 1;
125 % Control actions in external regulator
126 u_ctr = [4,5]';
127 % LQR external outputs
128 y_ctr = [1,2]';
129 % State space system external controller
130 A2 = A(x_ctr,x_ctr);
131 B2 = A(x_ctr,u_ctr);
132 C2 = C(y_ctr,x_ctr);
133 D2 = zeros(size(y_ctr,1),size(u_ctr,1));
134 % Extended external system
135 Ab2 = [A2 zeros(size(A2,1),size(C2,1)); -C2 zeros(size(C2,1),size(C2,1))];
136 Bb2 = [B2; zeros(size(C2,1),size(B2,2))];
137 % Extended system controllability
138 Co = ctrb(Ab2,Bb2);
139 unco = length(Ab2) - rank(Co);
140 fprintf('Uncontrollable states internal extended system: %g\n',unco)
141 % Q y R LQR Matrices
142 Q = blkdiag(0,0,0,0,0.07,0.07);
143 R = diag([1,1]);
144 % External LQR
145 [Kb,:,:] = lqr(Ab2,Bb2,Q,R);
146 K2b = Kb(:,1:(size(A2,2))); K2xy = K2b(2,1); K2uv = K2b(2,3);
147 Ki2b = -Kb(:,end-size(C2,1)+1:end); Ki2 = Ki2b(2,1);
148 fprintf('Extended external system poles:\n')
149 eig(Ab2-Bb2*Kb)
save('mat_controlposicion_params.mat','K1','K2b','Ki1','Ki2b','C1','C2',...
180 'Ck','u0','Mixer','x_ctr_2_1','x_ctr_2_2','max_omega','vk','x0',...
181 'Izz','Iyy','Ixx','m','g','invMixer','Pk0','Qk','Rk','K2xy','K2uv',...
182 'Ki2')