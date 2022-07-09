import numpy as np
import torch

class environment():
    def __init__(self,Cart_Params,device,dtype):
        self.L = Cart_Params['L']
        self.m = Cart_Params['m']
        self.M = Cart_Params['M']
        self.g = Cart_Params['g']
        self.dt = Cart_Params['dt']
        self.nu = Cart_Params['nu']
        self.C = Cart_Params['C']
        self.action_space = Cart_Params['action_space']

        self.num_states = 4
        self.num_controls = 2
        self.num_measurements = 2

        self.device = device
        self.dtype = dtype

    def step(self,x,u,xc):
        u = self.unorm_inputs(u)

        u = self.NLDI(x,u)

        cX = torch.cos(x[:,2])
        sX = torch.sin(x[:,2])

        xddot = (u[:,0] + u[:,1]*cX/self.L - self.m*sX*(self.L*x[:,3]**2 - \
              self.g*cX))/(self.M + self.m*(1 - cX**2))
        thetaddot = u[:,1]/self.m/self.L**2 + (cX*xddot + self.g*sX)/self.L

        xp = x + self.dt*torch.cat((x[:,1].view(-1,1),xddot.view(-1,1),
                                    x[:,3].view(-1,1),thetaddot.view(-1,1)),1)


        greater = xp[:,2] > np.pi
        lesser = xp[:,2] <= -np.pi
        if greater is not None:
            xp[greater,2] -= 2*np.pi
        if lesser is not None:
            xp[lesser,2] += 2*np.pi

        yp = self.measurement(xp)

        r = 0.01*(torch.sum((yp - xc)**2,1) + 0.001*torch.sum(u**2,1))

        return xp, r

    def unorm_inputs(self,u):
        return (self.action_space[1] - self.action_space[0])*(u + 1)/2\
                    + self.action_space[0]

    def NLDI(self,x,u):
        cX = torch.cos(x[:,2])
        sX = torch.sin(x[:,2])

        alpha = u[:,1] - (cX*u[:,0] + self.g*sX)/self.L
        tau = self.m*self.L**2*alpha
        f = (self.M + self.m*(1 - cX**2))*u[:,0] - cX/self.L*tau +\
            self.m*sX*(self.L*x[:,3]**2 - self.g*cX)

        return torch.cat((f.view(-1,1),tau.view(-1,1)),1)

    def measurement(self,x):
        N = len(x)
        return torch.sum(self.C.view(1,self.num_measurements,self.num_states)\
            *x.view(N,1,self.num_states),2) #+ self.nu*torch.randn((N,
            # self.num_measurements),device=self.device,dtype=self.dtype)

    def Jacobian(self,X,U,continuous_output=False):

        X = X[:,:-1]
        cX = torch.cos(X[:,:,2])
        sX = torch.sin(X[:,:,2])
        X3_2 = X[:,:,3]**2

        Nb = len(X)
        Nh = len(X[0])

        # xddot derivative
        alpha = U - self.m*sX*(self.L*X3_2 - self.g*cX)
        beta = self.M + self.m*(1 - cX**2)

        xddot = alpha/beta

        dalpha_dx2 = -self.m*cX*(self.L*X3_2 - self.g*cX) - self.g*sX**2*self.m
        dbeta_dx2 = 2*self.m*cX*sX

        dxddot_dx2 = (dalpha_dx2*beta - alpha*dbeta_dx2)/beta**2
        dxddot_dx3 = -2*self.L*X[:,:,3]*self.m*sX/beta

        dxddot_du = 1/beta

        # thetaddot derivative
        dthetaddot_dx2 = (cX*(self.g + dxddot_dx2) - sX*xddot)/self.L
        dthetaddot_dx3 = cX*dxddot_dx3/self.L

        dthetaddot_du = cX*dxddot_du/self.L

        # Euler integration
        I = torch.eye(4).view(1,1,4,4)*torch.ones((Nb,Nh,4,4))
        I = I.to(self.device).type(self.dtype)

        zeros = torch.zeros((Nb,Nh,1),device=self.device,dtype=self.dtype)
        ones = 1+zeros

        df_dx0 = torch.cat((zeros,zeros,zeros,zeros),2)
        df_dx1 = torch.cat((ones,zeros,zeros,zeros),2)
        df_dx2 = torch.cat((zeros,dxddot_dx2.unsqueeze(2),zeros,
                                                dthetaddot_dx2.unsqueeze(2)),2)
        df_dx3 = torch.cat((zeros,dxddot_dx3.unsqueeze(2),ones,
                                                dthetaddot_dx3.unsqueeze(2)),2)

        df_dx = torch.cat((torch.unsqueeze(df_dx0,3),
            torch.unsqueeze(df_dx1,3),torch.unsqueeze(df_dx2,3),
            torch.unsqueeze(df_dx3,3)),3)

        df_du = torch.cat((zeros,dxddot_du.view(Nb,Nh,1),zeros,
            dthetaddot_du.view(Nb,Nh,1)),2)

        dxp_dx = I + self.dt*df_dx
        dxp_du = self.dt*df_du

        if not continuous_output:
            return dxp_dx, dxp_du

        return dxp_dx, dxp_du, df_dx, df_du
