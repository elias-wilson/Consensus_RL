import torch

class CRCC():
    def __init__(self,Cont_Params,device,dtype):
        self.device = device
        self.dtype = dtype
        self.G = Cont_Params['G']
        self.H = Cont_Params['H']
        self.N = Cont_Params['N']
        self.F = Cont_Params['F']
        self.num_zeros = self.N - torch.sum(self.G,1)
        self.lr = Cont_Params['lr']
        self.max = Cont_Params['max']
        self.min = Cont_Params['min']
        self.row_ids = torch.arange(self.N,device=device,dtype=torch.long)\
            .view(-1,1)*torch.ones((self.N,self.N-self.F-1),
            device=self.device,dtype=torch.long)

    def step(self,z):
        B = len(z)

        # difference [[z0-z0,z0-z1,...,z0-zN],...,[zN-z0,...,zN-zN]]
        d =  z.view(B,self.N,1,-1).repeat(1,1,self.N,1) - \
                                    z.view(B,1,self.N,-1).repeat(1,self.N,1,1)

        # Apply the connectivity map (Fully connnected won't change anything)
        m = d*self.G.view(1,self.N,self.N,1)

        # distance (2-norm)
        n = torch.norm(d,p=2,dim=3)

        # find sorting indices to maniuplate the difference tensor
        _, sort_ids = torch.sort(n,dim=2)

        # Create indices to sort the difference tesnor
        batch_ids = torch.arange(B,device=self.device,dtype=torch.long)\
            .view(B,1,-1)*torch.ones((B,self.N,self.N-self.F-1),
            device=self.device,dtype=torch.long)
        col_ids = sort_ids[:,:,1:self.N-self.F] # Grab the parts we want


        # Filter the difference tensor
        d_filtered = d[batch_ids,self.row_ids,col_ids]

        # Find next estimate
        zp = z - self.lr*torch.sum(d_filtered,2)
        zp = self.projection(zp)

        return zp

    def projection(self,z):
        return torch.minimum(torch.maximum(z,self.min),self.max)
