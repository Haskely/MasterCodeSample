import torch as th

class DNN(th.nn.Module):
    def __init__(self, input_dim=3, hidden_layer_nodes=[16,16,32,32,64,64], output_dim=3, Activation = th.nn.Tanh) -> None:
        super().__init__()

        in_out_dims = [input_dim] + hidden_layer_nodes
        layers = []
        for i in range(len(in_out_dims) - 1):
            layers.append(th.nn.Linear(in_out_dims[i], in_out_dims[i+1]))
            layers.append(Activation())
        layers.append(th.nn.Linear(in_out_dims[-1], output_dim))
        self.dnn = th.nn.Sequential(*layers)

    def forward(self, input: th.Tensor):
        return self.dnn(input)

class ResidualBlock(th.nn.Module):
    def __init__(self, input_dim=64, hidden_layer_nodes=[64]*2, Activation = None) -> None:
        super().__init__()
        self.dnn = DNN(input_dim,hidden_layer_nodes,input_dim,Activation)
        self.activation = Activation()
        
    def forward(self, input: th.Tensor):
        return self.activation(self.dnn(input) + input)
    
class ResidualDNN(th.nn.Module):
    def __init__(self, input_dim=3, resblock_layer_nodes=[64] * 4, output_dim=3, Activation = None) -> None:
        super().__init__()
        
        layers = []
        layers.append(th.nn.Linear(input_dim, resblock_layer_nodes[0]))
        layers.append(Activation())
        for n in resblock_layer_nodes:
            layers.append(ResidualBlock(n,[n,n],Activation))
            
        layers.append(th.nn.Linear(resblock_layer_nodes[-1],output_dim))
        
        self.resdnn = th.nn.Sequential(*layers)

    def forward(self, input: th.Tensor):
        return self.resdnn(input)
    
class Net(th.nn.Module):
    def __init__(self, cared_tscoor: th.Tensor = None) -> None:
        super().__init__()
        self.cared_tscoor_mean = th.nn.Parameter(
            th.zeros((3)), requires_grad=False)
        self.cared_tscoor_std = th.nn.Parameter(
            th.ones((3)), requires_grad=False)
        self.set_normalize(cared_tscoor)
    
    def set_normalize(self,cared_tscoor = None):
        if cared_tscoor is not None:
            assert cared_tscoor.dim() == 2
            assert cared_tscoor.shape[1] == 3
            # 考虑是否要对空间维度等比例缩放，以便保持原几何性质
            # tscoor_max = cared_tscoor.max(dim=0)
            # tscoor_min = cared_tscoor.min(dim=0)
            mean = cared_tscoor.mean(dim=0)
            std = cared_tscoor.std(dim=0)
            self.cared_tscoor_mean = th.nn.Parameter(
                mean, requires_grad=False)
            self.cared_tscoor_std = th.nn.Parameter(
                std, requires_grad=False)
        
    def normalize_tscoor(self,tscoor):
        return (tscoor - self.cared_tscoor_mean)/(self.cared_tscoor_std+1e-06)
    
    def _calculate(self,tscoor:th.Tensor):
        raise NotImplementedError()

    def forward(self, tscoor):
        tscoor = self.normalize_tscoor(tscoor)
        return self._calculate(tscoor)
    
class StokesNet(Net):
    def __init__(self,cared_tscoor: th.Tensor = None,hidden_layer_nodes=[16,16,32,32,64,64]) -> None:
        super().__init__(cared_tscoor)
        # self.p0 = th.nn.Parameter(th.tensor(p0), requires_grad=False)
        self.dnn = DNN(hidden_layer_nodes=hidden_layer_nodes)
    
    def _calculate(self,tscoor:th.Tensor):
        return self.dnn(tscoor)
    
class DarcyNet(Net):
    def __init__(self,cared_tscoor: th.Tensor = None,hidden_layer_nodes=[16,16,32,32,64,64]) -> None:
        super().__init__(cared_tscoor)
        # self.p0 = th.nn.Parameter(th.tensor(p0), requires_grad=False)
        self.dnn = DNN(hidden_layer_nodes=hidden_layer_nodes)
    
    def _calculate(self,tscoor:th.Tensor):
        return self.dnn(tscoor)
    
class CombinedNet(th.nn.Module):
    def __init__(self,hidden_layer_nodes=[32]*8, ns_cared_tscoor = None, darcy_cared_tscoor = None):
        super().__init__()
        self.stokesNet = StokesNet(cared_tscoor = ns_cared_tscoor,hidden_layer_nodes=hidden_layer_nodes)
        self.darcyNet = DarcyNet(cared_tscoor = darcy_cared_tscoor,hidden_layer_nodes=hidden_layer_nodes)

    def NS_th(self,T: th.Tensor,X: th.Tensor,Y: th.Tensor):
        TXY = th.stack([T,X,Y],dim=-1)
        UVP:th.Tensor = self.stokesNet(TXY)
        return UVP.unbind(dim = -1)
    
    def Darcy_th(self, T: th.Tensor,X: th.Tensor,Y: th.Tensor):
        TXY = th.stack([T,X,Y],dim=-1)
        UVP:th.Tensor = self.darcyNet(TXY)
        return UVP.unbind(dim = -1)

class ResidualCombinedNet(th.nn.Module):
    def __init__(self,resblock_layer_nodes=[32] * 3, Activation=th.nn.Tanh):
        super().__init__()
        self.stokesNet = ResidualDNN(resblock_layer_nodes=resblock_layer_nodes, Activation=Activation)
        self.darcyNet = ResidualDNN(resblock_layer_nodes=resblock_layer_nodes, Activation=Activation)

    def NS_th(self,T: th.Tensor,X: th.Tensor,Y: th.Tensor):
        TXY = th.stack([T,X,Y],dim=-1)
        UVP:th.Tensor = self.stokesNet(TXY)
        return UVP.unbind(dim = -1)
    
    def Darcy_th(self, T: th.Tensor,X: th.Tensor,Y: th.Tensor):
        TXY = th.stack([T,X,Y],dim=-1)
        UVP:th.Tensor = self.darcyNet(TXY)
        return UVP.unbind(dim = -1)

if __name__ == '__main__':
    from torchinfo import summary
    rescombinednet = ResidualCombinedNet()
    print(summary(rescombinednet))