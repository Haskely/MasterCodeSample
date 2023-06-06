import torch as th
from network import CombinedNet, ResidualCombinedNet
from examples import Example_Longtime2012
from pathlib import Path 

def L2(tensor:th.Tensor):
    return tensor.square().sum().sqrt()

def L1(tensor:th.Tensor):
    return tensor.abs().sum()

def test():
    exam = Example_Longtime2012
    
    output_dir = Path(r'output\train_Example_Longtime2012_CombinedNet_32x6_inverse1__2022-05-14 23h33m05s')
    trained_combinedNet = CombinedNet([32]*6)
    state_fp = output_dir / 'nsd_net_state.pth'
    trained_combinedNet.load_state_dict(th.load(state_fp, map_location=th.device('cpu')))
    # 3D 绘制尝试2
    gridn = 20
    T_f, X_f, Y_f = th.meshgrid(th.linspace(0, 1, gridn), th.linspace(0, 1, gridn*3), th.linspace(1, 2, gridn*3))
    T_p, X_p, Y_p = th.meshgrid(th.linspace(0, 1, gridn), th.linspace(0, 1, gridn*3), th.linspace(0, 1, gridn*3))

    true_U_f,true_V_f,true_P_f = exam.NS_th(T_f,X_f,Y_f)
    true_U_p,true_V_p,true_P_p = exam.Darcy_th(T_p,X_p,Y_p)
    pred_U_f,pred_V_f,pred_P_f = [x.detach() for x in trained_combinedNet.NS_th(T_f,X_f,Y_f)]
    pred_U_p,pred_V_p,pred_P_p = [x.detach() for x in trained_combinedNet.Darcy_th(T_p,X_p,Y_p)]
    
    trueA = th.concat([true_U_f,true_V_f,true_P_f,true_U_p,true_V_p,true_P_p])
    predA = th.concat([pred_U_f,pred_V_f,pred_P_f,pred_U_p,pred_V_p,pred_P_p])
    relative_errorA = L2(predA - trueA) / L2(trueA)
    print(f'relative_errorA: {relative_errorA}')
    
    rela_errs = []
    for t,p in zip([true_U_f,true_V_f,true_P_f,true_U_p,true_V_p,true_P_p],[pred_U_f,pred_V_f,pred_P_f,pred_U_p,pred_V_p,pred_P_p]):
        rela_errs.append(L2(p - t) / L2(t))
        
    
    print(f'mean_rela_err: {th.mean(th.tensor(rela_errs))}')
    print(rela_errs)
    
    print(f'{th.mean(th.tensor(rela_errs[0:2])):6f}')
    print(f'{th.mean(th.tensor(rela_errs[3:5])):6f}')
    print(f'{th.mean(th.tensor([rela_errs[2],rela_errs[5]])):6f}')
test()
        