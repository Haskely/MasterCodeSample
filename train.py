
import time
from pathlib import Path

from sympy import false
from PhysicalProcessDataClass import PhysicalProcessData as PPD
from network import CombinedNet
from examples import create_TXY_by_range
from loss import setup_exam2eqns
from network import CombinedNet
from loss import navier_stokes_eqns, darcy_eqns, interface_eqns
import torch as th
from torch.utils.tensorboard import SummaryWriter
import time
from pathlib import Path
from tqdm.notebook import tqdm
import numpy as np
from PhysicalProcessDataClass import combine

def get_ppd4phy(ppd:PPD, init_t=0.0, x_range=(0, 1), y_range=(0, 1), d=0.01, is_stable = False):
    x1, x2 = x_range
    y1, y2 = y_range
    
    boundary_ppd = ppd.cutRangeData(dict(X=[x1 + d, x2 - d], Y=[y1 + d, y2 - d]))
    boundary_ppd.name = ppd.name + ' 边值条件'
    if is_stable:
        return boundary_ppd
    else:
        return combine(ppd.snapshot(t=init_t), boundary_ppd, name=ppd.name + ' 初边值条件')

def get_ppd4phy_circle4inverse(ppd:PPD, center=(0.5,0.5), radius=0.25):
    cen_x,cen_y = center
    
    keeped_ppd = ppd.subData(lambda row: ((row["X"]-cen_x)**2 + (row["Y"]-cen_y)**2 <= radius**2))
    keeped_ppd.name = ppd.name + '圆形切片'
    return keeped_ppd


def prepare_train(exam,combinedNet:CombinedNet,is_inverse = False,commit = 'fd'):
    
    output_dir = f'./output/train_{exam.__name__}_{combinedNet.__class__.__name__}_{commit}_{time.strftime("_%Y-%m-%d %Hh%Mm%Ss")}'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    from examples import example2ppd
    stokes_ppd, darcy_ppd = example2ppd(exam)

    

    combine(stokes_ppd, darcy_ppd).plot_snapshot(
        save_dir=output_dir, title='读取到的数据')

    ns_ppd4eqn = stokes_ppd.subPhysicalQuantity([])
    ns_ppd4eqn.name = 'Stokes 方程损失计算'

    darcy_ppd4eqn = darcy_ppd.subPhysicalQuantity([])
    darcy_ppd4eqn.name = 'Darcy 方程损失计算'

    if not is_inverse:
        ns_ppd4phy = get_ppd4phy(stokes_ppd, y_range=(0.98, 2),is_stable=(exam.NS_area['Tn'] <= 1))
        darcy_ppd4phy = get_ppd4phy(darcy_ppd, y_range=(0, 1.02),is_stable=(exam.Darcy_area['Tn'] <= 1))
    else:
        ns_ppd4phy = get_ppd4phy_circle4inverse(stokes_ppd, center=(0.5,1.5))
        darcy_ppd4phy = get_ppd4phy_circle4inverse(darcy_ppd, center=(0.5,0.5))


    Ti, Xi, Yi = [x.detach().cpu().numpy()
                  for x in create_TXY_by_range(**exam.Interface_area)]
    interface_ppd4eqn = PPD(name='Interface 方程损失计算', T=Ti, X=Xi, Y=Yi)


    train_kwargs = dict(
        ns_ppd4phy=ns_ppd4phy,
        ns_ppd4eqn=ns_ppd4eqn,
        darcy_ppd4phy=darcy_ppd4phy,
        darcy_ppd4eqn=darcy_ppd4eqn,
        interface_ppd4eqn=interface_ppd4eqn,
        batch_size=10240,
        output_dir=output_dir,
    )
    
    from torchinfo import summary
    net_desc = summary(combinedNet)
    with open(Path(output_dir) / 'info.txt','w') as f:
        f.write(f"batchsize:{train_kwargs['batch_size']}\n\n")
        f.write('='*5 + 'Exam_Summary' + '='*5)
        f.write(exam.Info + '\n\n')
        f.write('='*5 + 'Net_Summary' + '='*5)
        f.write(str(net_desc) + '\n\n')
        for ppd,name in zip((combine(ns_ppd4phy, darcy_ppd4phy),combine(ns_ppd4eqn, darcy_ppd4eqn),interface_ppd4eqn),('物理训练数据','方程训练数据(S-D)','方程训练数据(交界面)')):
            f.write(name + '\n')
            f.write(str(ppd.describe()) + '\n')
            
            ppd.plot_snapshot(save_dir=output_dir, title=name)
        
    return train_kwargs


def _check_ppd(ppd: PPD, ppd_type: str = 'tscoor', TimeSpaceNames=("T", "X", "Y"), PhysicalQuantityNames=('U', 'V', 'P')):
    assert ppd.TimeSpaceNames == TimeSpaceNames
    check_assert = {
        'tscoor': len(ppd.PhysicalQuantityNames) == 0,
        'uvp': set(ppd.PhysicalQuantityNames).issubset(set(PhysicalQuantityNames)),
    }
    assert ppd_type in check_assert
    assert check_assert[ppd_type]

class ValueHelper:
    def __init__(self) -> None:
        self.name2values = dict()

    def add_value(self, name: str, value):
        if name not in self.name2values:
            self.name2values[name] = list()
        self.name2values[name].append(value)

    def get_mean(self, name: str):
        return np.mean(self.name2values[name])

    def clean(self, name):
        self.name2values[name] = list()

def train(
        combinedNet: CombinedNet,
        eqnloss_funcs: dict(stokes=None, darcy=None, interface=None),
        ns_ppd4phy: PPD,
        ns_ppd4eqn: PPD,
        darcy_ppd4phy: PPD,
        darcy_ppd4eqn: PPD,
        interface_ppd4eqn: PPD,
        batch_size: int,
        output_dir: str = './output'):

    for ppd in (ns_ppd4phy, darcy_ppd4phy):
        _check_ppd(ppd, 'uvp')
    for ppd in (ns_ppd4eqn, darcy_ppd4eqn, interface_ppd4eqn):
        _check_ppd(ppd, 'tscoor')

    output_dir: Path = Path(output_dir)
    if output_dir.exists():
        output_dir.with_name(
            output_dir.name + time.strftime("_%Y-%m-%d %Hh%Mm%Ss"))
    output_dir.mkdir(exist_ok=True, parents=True)
    board_writer = SummaryWriter(log_dir=output_dir / 'tensorboard')
    print(
        f'执行: tensorboard --logdir="{Path(board_writer.log_dir).resolve()}"')

    DEVICE = th.device("cuda" if th.cuda.is_available() else "cpu")
    combinedNet.to(DEVICE)
    board_writer.add_text('log', f'在 {DEVICE} 上执行')

    LR = 1e-2
    MIN_LR = LR * 1e-3

    LR_STEP_N = 10
    optimizer = th.optim.Adam(combinedNet.parameters(), lr=LR)

    lr_scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, min_lr= 0.0, eps=0.0)
    lr_scheduler2 = th.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.5)
    
    board_writer.add_text(
        'log', f"batch_size:{batch_size},初始lr:{optimizer.param_groups[0]['lr']},MIN_LR:{MIN_LR},LR_STEP_N:{LR_STEP_N}")

    valuehelper = ValueHelper()
    bar = tqdm()

    IterN = 1
    lr = optimizer.param_groups[0]['lr']
    last_reduce_lr_t = time.time()

    def phyloss4onebatch(ppd4phy:PPD, net, batch_size):
        batch_size = min(len(ppd4phy.DataDF),batch_size)
        TXYUVP = th.from_numpy(ppd4phy.DataDF.sample(
            batch_size).to_numpy(dtype='float32')).to(DEVICE)
        TXY = TXYUVP[:, 0:3]
        true_UVP = TXYUVP[:, 3:6]
        pred_UVP = net(TXY)
        
        true_pn_i = {pn:i for i,pn in enumerate(ppd4phy.DataDF.columns[3:])}
        loss_dict = {}
        for i,pn in enumerate(['U','V','P']):
            if pn in true_pn_i:
                ti = true_pn_i[pn]
                loss_dict[f'{pn}_loss'] = (10**3) * (pred_UVP[:, i] - true_UVP[:, ti]).square().mean()
            else:
                loss_dict[f'{pn}_loss'] = th.tensor(0.0)
        return loss_dict

    def eqnloss4onebatch(ppd4eqn, eqnloss_func, func_kwargs, batch_size):
        batch_size = min(len(ppd4eqn.DataDF),batch_size)
        TXY = th.from_numpy(ppd4eqn.DataDF.sample(
            batch_size).to_numpy(dtype='float32')).to(DEVICE)
        T, X, Y = TXY.unbind(dim=-1)

        return (_loss.abs().mean(dim=0).sum() for _loss in eqnloss_func(T, X, Y, **func_kwargs))

    while lr > MIN_LR:

        optimizer.zero_grad()

        ns_phyloss_dict = phyloss4onebatch(
            ns_ppd4phy, combinedNet.stokesNet, batch_size)
        darcy_phyloss_dict = phyloss4onebatch(
            darcy_ppd4phy, combinedNet.darcyNet, batch_size)

        ns_eqnloss_dict = {k: v for k, v in zip(
            ('ns_eqn1', 'ns_eqn2'),
            eqnloss4onebatch(ns_ppd4eqn,
                             eqnloss_funcs['stokes'],
                             dict(stokes_UVP_func=combinedNet.NS_th),
                             batch_size))}
        darcy_eqnloss_dict = {k: v for k, v in zip(
            ('darcy_eqn1', 'darcy_eqn2'),
            eqnloss4onebatch(darcy_ppd4eqn,
                             eqnloss_funcs['darcy'],
                             dict(darcy_UVP_func=combinedNet.Darcy_th),
                             batch_size))}

        interface_eqnloss_dict = {k: v for k, v in zip(
            ('inter_eqn1', 'inter_eqn2', 'inter_eqn3'),
            eqnloss4onebatch(interface_ppd4eqn,
                             eqnloss_funcs['interface'],
                             dict(stokes_UVP_func=combinedNet.NS_th,
                                  darcy_UVP_func=combinedNet.Darcy_th),
                             batch_size))}

        loss = sum([sum(lossdict.values()) for lossdict in (ns_phyloss_dict,
                                                            darcy_phyloss_dict,
                                                            ns_eqnloss_dict,
                                                            darcy_eqnloss_dict,
                                                            interface_eqnloss_dict)])

        loss.backward()
        optimizer.step()

        board_writer.add_scalar(
            'TotalLoss', loss.item(), global_step=IterN)
        board_writer.add_scalar('lr', lr, global_step=IterN)
        for tag, lossdicts in zip(['Stokes', 'Darcy'], [(ns_phyloss_dict, ns_eqnloss_dict), (darcy_phyloss_dict, darcy_eqnloss_dict)]):
            for tyn, ld in zip(('phyloss', 'eqnloss'), lossdicts):
                for name, value in ld.items():
                    board_writer.add_scalar(
                        f'{tag}/{tyn}-{name}', value.item(), global_step=IterN)
        for name, value in interface_eqnloss_dict.items():
            board_writer.add_scalar(
                f'Interface/eqnloss-{name}', value.item(), global_step=IterN)

        valuehelper.add_value('loss', loss.item())
        if IterN % LR_STEP_N == 0:
            mean_loss = valuehelper.get_mean('loss')
            valuehelper.clean('loss')

            if time.time() - last_reduce_lr_t > 30 * 60:
                lr_scheduler2.step()
                lr_scheduler._reset()
                board_writer.add_text('log','连续30minlr不变，强制降低lr')
                
            lr_scheduler.step(mean_loss)

            board_writer.add_scalar(
                f"mean_loss_{LR_STEP_N}-steps", mean_loss, global_step=IterN)

            th.save(combinedNet.state_dict(),
                    output_dir / 'nsd_net_state.pth')
            
        pre_lr = lr
        lr = optimizer.param_groups[0]['lr']
        if pre_lr > lr:
            last_reduce_lr_t = time.time()
            board_writer.add_text('log',f'lr降低了，last_reduce_lr_t:{last_reduce_lr_t}')
            
        bar.set_description(
            desc=f"IterN:{IterN} Loss:{loss.item():g} lr:{lr:g}")

        bar.update()
        IterN += 1
        


