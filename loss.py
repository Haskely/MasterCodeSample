from typing import Dict
import torch as th
if th.cuda.is_available():
    th.set_default_tensor_type('torch.cuda.FloatTensor')
def grad4model(x: th.Tensor, y: th.Tensor) -> th.Tensor:
    """简介:输入代表自变量的张量 x，代表因变量的张量 y，输出 dy/dx, 并且该输出可以继续进行反向传播。
        这里自变量为 p 维向量，因变量为 q 维向量。但是注意，
        在神经网络训练中，往往一次输入 N 个 自变量 作为一个批次(batch),以便进行并行加速。
        若自变量原为 p 维向量，此时对应的输入 x 形状为 N x p，对应的输出 y 为 N x q。
        此时返回的是一个形状为 N x p x q 的三维张量，其中位置为 [i,j,k] 的元素代表 一个批次（N组数据）中 第i个自变量 的第 j 个分量 对于 第i个因变量 的第 k 个 分量的偏导数。

    Args:
        x (th.Tensor): 代表自变量的x（为了并行运算，形状可能为 N x p）
        y (th.Tensor): 代表因变量的y（为了并行运算，形状可能为 N x q）

    Returns:
        th.Tensor: 输出 dy/dx, 一个形状为 N x p x q 的三维张量，其中位置为 [i,j,k] 的元素代表 一个批次（N组数据）中 第i个自变量 的第 j 个分量 对于 第i个因变量 的第 k 个 分量的偏导数。
    """
    
    assert x.shape[0] == y.shape[0], f'根据本函数的设计，并行数量 N 应该一致，但是 x:{x.shape} 与 y:{y.shape} 的数量维度 N 不一致'
    assert y.dim() == 2 or y.dim() == 1, f'代表因变量的张量 y 的形状应该为 N x q 的二维张量，但是收到y维度为 {y.dim()} != 2'
    
    if y.dim() == 2:
        _y = y.sum(dim=0)  # 1 x q
        grad_res = th.nan_to_num(th.stack([th.autograd.grad(_y[i], x, create_graph=True, allow_unused=True)[
                            0] for i in range(y.shape[1])], dim=-1))
    elif y.dim() == 1:
        _y = y.sum(dim=0)  # 标量
        grad_res = th.autograd.grad(_y, x, create_graph=True, allow_unused=True)[0]
        if grad_res is None:
            grad_res = 0.0
    return grad_res

def navier_stokes_eqns(T: th.Tensor,X: th.Tensor,Y: th.Tensor, stokes_UVP_func, input_f = lambda *args:[0.0,0.0], is_nonliner_type = True, is_stable = False) -> th.Tensor:
    """
    输入一组时空坐标 TXY N x 3,
    输入一个torch写的函数 stokes_UVP_func，输入 TXY,输出UVP
    
    输出 Stokes 方程组损失 N x 2,N x 1，分别为第一二个方程
    """
    [x.requires_grad_() for x in [T,X,Y]]
    
    U,V,P = stokes_UVP_func(T,X,Y)


    Ux = grad4model(X,U)
    Uy = grad4model(Y,U)
    Vx = grad4model(X,V)
    Vy = grad4model(Y,V)

    Uxx = grad4model(X,Ux)
    Uyy = grad4model(Y,Uy)
    Vxx = grad4model(X,Vx)
    Vyy = grad4model(Y,Vy)

    rho = 1.0
    Nu = 1.0
    
    f_1,f_2 = input_f(T,X,Y)

    Px,Py = grad4model(X,P),grad4model(Y,P)
    
    if is_stable:
        Ut = 0.0
        Vt = 0.0
    else:
        Ut = grad4model(T,U)
        Vt = grad4model(T,V)
        
    if is_nonliner_type:
        eqn1_x = rho * (Ut + U * Ux + V * Uy) -Nu * (Uxx + Uyy) + Px - f_1 # N x 1
        eqn1_y = rho * (Vt + U * Vx + V * Vy) -Nu * (Vxx + Vyy) + Py - f_2 # N x 1
    else:
        eqn1_x = rho * Ut -Nu * (Uxx + Uyy) + Px - f_1 # N x 1
        eqn1_y = rho * Vt -Nu * (Vxx + Vyy) + Py - f_2 # N x 1
    eqn1 = th.stack([eqn1_x,eqn1_y],dim=-1) # N x 2
    
    eqn2 = Ux + Vy# N x 1
    
    return eqn1,eqn2

def darcy_eqns(T: th.Tensor,X: th.Tensor,Y: th.Tensor, darcy_UVP_func, input_f = lambda *args:[[0.0],0.0], is_stable = False):
    """    
    输入一组时空坐标 TXY N x 3,
    输入一个torch写的函数 darcy_UVP_func 输入TXY,输出UVP
    
    输出 Darcy 方程组损失 N x 2,N x 2，分别为第一二个方程
    """
    [x.requires_grad_() for x in [T,X,Y]]
    
    U,V,P = darcy_UVP_func(T,X,Y)
    
    Ux = grad4model(X,U)
    # Uy = grad4model(Y,U)
    # Vx = grad4model(X,V)
    Vy = grad4model(Y,V)

    nabla_P = th.stack([grad4model(X,P),grad4model(Y,P)]).T
    
    mu = 1.0
    kappa_inv = th.eye(2)
    beta = 0.0
    rho = 1.0

    
    S0 = 1.0 
    
    f1,f2 = input_f(T,X,Y)
    
    UV = th.stack([U,V]).T # N x 2
    
    if is_stable:
        Pt = 0.0
    else:
        # Ut = grad4model(T,U)
        # Vt = grad4model(T,V)
        Pt = grad4model(T,P)
    
    eqn1 = mu * UV @ kappa_inv + beta * UV.norm(dim=1, keepdim=True) * UV + rho * (nabla_P - f1) # N x 2
    eqn2 = S0 * Pt + Ux + Vy - f2 # N x 1
    
    return eqn1,eqn2

def interface_eqns(T: th.Tensor,X: th.Tensor,Y: th.Tensor, stokes_UVP_func, darcy_UVP_func):
    """    输入一组时空坐标 TXY N x 3,
    输入一个torch写的函数 stokes_UVP_func，输入TXY,输出UVP
    输入一个torch写的函数 darcy_UVP_func 输入TXY,输出UVP
    
    输出 交界面 方程组损失 N x 1,N x 1，N x 1 分别为第一二三个方程

    Args:
        TXY (th.Tensor): _description_
        stokes_UVP_func (_type_): _description_
        darcy_UVP_func (_type_): _description_

    Returns:
        _type_: _description_
    """
    [x.requires_grad_() for x in [T,X,Y]]
    
    Us,Vs,Ps = stokes_UVP_func(T,X,Y) # N x 3
    Ud,Vd,Pd = darcy_UVP_func(T,X,Y) # N x 3
    
    n_S = th.Tensor([0.0,-1.0]) # Stokes区域的外法向量
    
    eqn1 = th.stack([Us,Vs],dim=-1) @ n_S - th.stack([Ud,Vd],dim=-1) @ n_S # N x 1
    
    mu = 1.0
    Ux = grad4model(X,Us)
    Uy = grad4model(Y,Us)
    Vx = grad4model(X,Vs)
    Vy = grad4model(Y,Vs)
    
    nabla_UVs = th.stack([th.stack([Ux,Vx],dim = -1),th.stack([Uy,Vy],dim = -1)],dim = -1) # N x 2 x 2
    
    DD_UVs_x2 = nabla_UVs + nabla_UVs.transpose(dim0=1,dim1=2)

    eqn2 = Ps - mu * (n_S @ DD_UVs_x2 @ n_S) - Pd # N x 1            n_S @ UV_diff_XY @ n_S
    
    t = th.Tensor([1.0,0.0]) # 边界的切向量
    G = 1.0
    UVs = th.stack([Us,Vs]).T # N x 2
    UVd = th.stack([Ud,Vd]).T # N x 2
    eqn3 = - mu * (t @ DD_UVs_x2 @ n_S) - G * (UVs - UVd) @ t # N x 1
    
    return eqn1,eqn2,eqn3


    

from examples import create_TXY_by_range,Example_Longtime2012

def setup_exam2eqns(exam:Example_Longtime2012):
    def _navier_stokes_eqns(T: th.Tensor,X: th.Tensor,Y: th.Tensor, stokes_UVP_func):
        return navier_stokes_eqns(T,X,Y, stokes_UVP_func, input_f=exam.NS_input_f, is_stable=(exam.NS_area['Tn'] <= 1))
    
    def _darcy_eqns(T: th.Tensor,X: th.Tensor,Y: th.Tensor, darcy_UVP_func):
        return darcy_eqns(T,X,Y,darcy_UVP_func,input_f = exam.Darcy_input_f, is_stable=(exam.Darcy_area['Tn'] <= 1))
    
    return _navier_stokes_eqns,_darcy_eqns,interface_eqns

def vertify_lossfuncs(exam:Example_Longtime2012, stokes_UVP_func,darcy_UVP_func):
    Ts,Xs,Ys = create_TXY_by_range(**exam.NS_area)
    Td,Xd,Yd = create_TXY_by_range(**exam.Darcy_area)
    Ti,Xi,Yi = create_TXY_by_range(**exam.Interface_area)
    
    navier_stokes_eqns,darcy_eqns,interface_eqns = setup_exam2eqns(exam)

    eqn_stokes_1,eqn_stokes_2 = navier_stokes_eqns(Ts,Xs,Ys,stokes_UVP_func)
    eqn_darcy_1,eqn_darcy_2 = darcy_eqns(Td,Xd,Yd,darcy_UVP_func)
    eqn_inter_1,eqn_inter_2,eqn_inter_3 = interface_eqns(Ti,Xi,Yi,stokes_UVP_func,darcy_UVP_func)
    
    eqn_dict = {
        "eqn_stokes_1":eqn_stokes_1,
        "eqn_stokes_2":eqn_stokes_2,
        "eqn_darcy_1":eqn_darcy_1,
        "eqn_darcy_2":eqn_darcy_2,
        "eqn_inter_1":eqn_inter_1,
        "eqn_inter_2":eqn_inter_2,
        "eqn_inter_3":eqn_inter_3
    }
    return eqn_dict