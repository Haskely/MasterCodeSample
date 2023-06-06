from PhysicalProcessDataClass import combine,read_DataDF
from examples import create_ppd,example2ppd,Example_Longtime2012
from network import CombinedNet
from loss import vertify_lossfuncs
import torch as th

def plot_diff(exam:Example_Longtime2012,trained_combinedNet:CombinedNet,savedir:'str'):
    stokes_ppd,darcy_ppd = example2ppd(exam)
    pred_stokes_ppd,pred_darcy_ppd = create_ppd(
        'pred_' + exam.__name__,
        exam.NS_area,exam.Darcy_area,
        trained_combinedNet.NS_th,trained_combinedNet.Darcy_th)

    SD_ppd = combine(stokes_ppd,darcy_ppd)
    SD_ppd.plot(save_dir=savedir, title='真实的数据')

    pred_SD_ppd = combine(pred_stokes_ppd,pred_darcy_ppd)
    pred_SD_ppd.plot(save_dir=savedir, title='预测的数据')

    diff_DF = SD_ppd.DataDF.copy()
    diff_DF[['U','V','P']] = (SD_ppd.DataDF - pred_SD_ppd.DataDF)[['U','V','P']]
    diff_ppd = read_DataDF(diff_DF,name='真解与预测解之差')
    diff_ppd.plot(save_dir=savedir)

def test_loss(exam:Example_Longtime2012,trained_combinedNet:CombinedNet):
    stokes_UVP_func = trained_combinedNet.NS_th 
    darcy_UVP_func = trained_combinedNet.Darcy_th

    eqn_dict = vertify_lossfuncs(exam,stokes_UVP_func,darcy_UVP_func)
    eqnloss_dict = {k:v.abs().mean(dim=0).sum() for k,v in eqn_dict.items()}
    
    stokes_ppd,darcy_ppd = example2ppd(exam)
    pred_stokes_ppd,pred_darcy_ppd = create_ppd(
        'pred_' + exam.__name__,
        exam.NS_area,exam.Darcy_area,
        trained_combinedNet.NS_th,trained_combinedNet.Darcy_th)
    
    phyloss = dict()
    
    for true_ppd,pred_ppd,name in zip([stokes_ppd,darcy_ppd],[pred_stokes_ppd,pred_darcy_ppd],['stokes','darcy']):
        phy_names = ['U','V','P']
        diffUVP:th.Tensor = th.from_numpy(pred_ppd.DataDF[phy_names].to_numpy()) - th.from_numpy(true_ppd.DataDF[phy_names].to_numpy())
        _loss = diffUVP.square().mean(dim = 0)
        for i,pn in enumerate(phy_names):
            phyloss[f'{name}_{pn}_loss'] = _loss[i]

    for name, loss in dict(**eqnloss_dict, **phyloss).items():
        print(f'{name}: {loss:g}')
    
    total_loss = sum(phyloss.values()) + sum(eqnloss_dict.values())
    print(f'\nTotalLoss:{total_loss:g}')