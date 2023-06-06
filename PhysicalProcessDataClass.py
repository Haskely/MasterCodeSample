
import math
import pandas as pd
import os
from pathlib import Path 
from typing import Callable, Dict, List, Tuple, Union
from matplotlib import colorbar
import numpy as np
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）

FigSubSize = 3

class PhysicalProcessData:

    def __init__(self, name: str = 'a_physicalProcessData', T: np.ndarray = np.zeros(1), X: np.ndarray = np.zeros(1), Y: np.ndarray = None, Z: np.ndarray = None, **PhysicalQuantity_ndarray_dict: np.ndarray) -> None:
        """初始化【物理过程数据】

        Args:
            T_array (np.ndarray, optional): [description]. 一维numpy向量，时间维度坐标。Defaults to np.zeros(1).
            X_array (np.ndarray, optional): [description]. 一维numpy向量，空间维度1坐标。Defaults to np.zeros(1).
            Y_array (np.ndarray, optional): [description]. 一维numpy向量，空间维度2坐标。Defaults to None.
            Z_array (np.ndarray, optional): [description]. 一维numpy向量，空间维度3坐标。Defaults to None.
            **PhysicalQuantity_ndarray_dict (dict[str,np.ndarray]): key为物理量名称，value为一维numpy向量，物理量数值。
        """
        self.name = name

        self.DataDF = pd.DataFrame()
        self.DataDF.loc[:,"T"] = T
        self.DataDF.loc[:,"X"] = X
        if Y is not None:
            self.DataDF.loc[:,"Y"] = Y
        if Z is not None:
            self.DataDF.loc[:,"Z"] = Z
    
        for pq_name, pq_array in PhysicalQuantity_ndarray_dict.items():
            while pq_name in self.DataDF.columns:
                pq_name = pq_name + "_"
            self.DataDF.loc[:,pq_name] = pq_array

        self._init_names()

    
    def _init_names(self):
        columns_set = set(self.DataDF.columns)

        self.TimeNames = ("T",)
        self.SpaceNames = ("X","Y","Z")[:len({"X","Y","Z"} & columns_set)]
        self.TimeSpaceNames = self.TimeNames + self.SpaceNames
        self.PhysicalQuantityNames = tuple(columns_set - set(self.TimeSpaceNames))

    def subPhysicalQuantity(self,PhysicalQuantityNames:Union[List,Tuple]):
        return PhysicalProcessData(self.name + "_subpqed",**self.DataDF[list(self.TimeSpaceNames+tuple(PhysicalQuantityNames))].to_dict("series"))


    def subData(self, whether2keep_func: Callable):
        """根据传入的函数判断限制返回自身的一个【物理过程数据】子集

        Args:
            whether2keep_func (function): 一个函数，传入自身DataDF的某一行判断是否保留该行（一个时空坐标）数据点
                例如：
                    lambda row: (row["X"]**2 + row["Y"]**2 <= 4)
        """

        return PhysicalProcessData(**self.DataDF[self.DataDF.apply(whether2keep_func, axis=1, result_type="reduce")].to_dict("series"))

    def cutData(self, whether2cut_func: Callable):
        """根据传入的函数判断限制切走自身的一个【物理过程数据】子集，剩下的数据保留下来

        Args:
            whether2cut_func (Callable): 一个函数，传入自身DataDF的某一行判断是否切走该行（一个时空坐标）数据点
                例如：
                    lambda row: (row["X"]**2 + row["Y"]**2 <= 4)

        Returns:
            [type]: [description]
        """        
        return self.subData(lambda row: ~whether2cut_func(row))

    def _range2cond(self, Ranges_dict: Dict[str, tuple]):
        cond = True
        for k, k_range in Ranges_dict.items():
            cond &= (k_range[0] < self.DataDF[k]) & (
                self.DataDF[k] < k_range[1])
        return cond

    def subRangeData(self, Ranges_dict: Dict[str, tuple]):
        """根据传入的范围限制返回自身的一个【物理过程数据】子集

        Args:
            Ranges_dict (dict[str,tuple]): 范围限制字典
                如{"T":[1,3],"X":[-1,1]}，将会保留该范围的数据

        Returns:
            (PhysicalProcessData): 符合传入限制的【物理过程数据】子集
        """

        return PhysicalProcessData(name = self.name + '_subranged',**self.DataDF[self._range2cond(Ranges_dict)].to_dict("series"))

    def cutRangeData(self, Ranges_dict: Dict[str, tuple]):
        """在空间中切走一个矩形，剩下的数据作为新的数据，返回新的PhysicalProcessData对象

        Args:
            Ranges_dict (dict[str,tuple]): 范围限制字典
                如{"T":[1,3],"X":[-1,1]}，将会切走该范围的数据

        Returns:
            PhysicalProcessData: 新的PhysicalProcessData对象
        """        
        return PhysicalProcessData(name = self.name + '_cutranged',**self.DataDF[~self._range2cond(Ranges_dict)].to_dict("series"))

    def snapshot(self, t:float = 0.0):
        """返回某一个时间切片

        Args:
            t (float, optional): 贴片的时间点，会自动找出距该时刻最近的时间坐标进行绘图. Defaults to 0.0.

        Returns:
            [type]: [description]
        """        
        t = self.DataDF["T"].iloc[(self.DataDF["T"]-t).abs().argsort().iloc[0]]
        return PhysicalProcessData(name = self.name + f'_snapshot(t={t:.3f})',**self.DataDF[self.DataDF["T"] == t].to_dict("series"))

    def plot(self, save_dir='', title: str = None, down_k: int = 1, time_frames: int = 50, save_type='gif'):
        """将物理过程数据制作成动态图或视频等，取决于传入的save_type

        Args:
            save_dir (str, optional): 保存到文件夹路径. Defaults to ''.
            title (str, optional): 动图标题. Defaults to None.
            down_k (int, optional): 空间下采样比率. Defaults to 1.
            time_frames (int, optional): 时间帧个数. Defaults to 50.
            save_type (str, optional): 保存类型. Defaults to 'gif'. 

        Returns:
            None
        """
        if title is None:
            title = f'{self.name}_animate'
        DataDFGroupedbyTime = self.DataDF.groupby("T")
        total_t_num = len(DataDFGroupedbyTime)
        if time_frames > total_t_num:time_frames = total_t_num

        SpaceNames = list(self.SpaceNames)
        PhysicalQuantityNames = list(self.PhysicalQuantityNames)
        pq_num = len(PhysicalQuantityNames)
        if time_frames <= 1 or pq_num == 0: return self.plot_snapshot(save_dir = save_dir,title = title)
        selected_times = tuple(DataDFGroupedbyTime.groups.keys())[0:total_t_num:round(total_t_num/time_frames)]
        selected_groups = [(t,DataDFGroupedbyTime.get_group(t)) for t in selected_times]

        ncols = math.ceil(math.sqrt(pq_num))
        nrows = math.ceil(pq_num/ncols)
        fig = plt.figure(figsize=(FigSubSize*ncols, FigSubSize*nrows))
        fig.set_tight_layout(True)
        fig.suptitle(title)
        axes_list = []
        colorbar_list = []
        vminmax_d = {}
        s = None
        def init():
            pj = None if len(SpaceNames) < 3 else '3d'
            for i in range(pq_num):
                axes = fig.add_subplot(nrows, ncols, i+1, projection=pj)
                axes_list.append(axes)
                pq_name = PhysicalQuantityNames[i]
                vminmax_d[pq_name] = (
                    self.DataDF[pq_name].min(), self.DataDF[pq_name].max())
                colorbar_list.append(fig.colorbar(axes.scatter(*[[]]*len(SpaceNames), c=[
                ], cmap=plt.hot(), vmin=vminmax_d[pq_name][0], vmax=vminmax_d[pq_name][1],s=s), ax=axes))


        def update(args):
            (t,oneTDataDF) = args
            point_num = oneTDataDF.shape[0]
            # step = round(point_num/down_num)
            step = down_k
            oneTDataDF = oneTDataDF.iloc[0:point_num:step]
            for i in range(pq_num):
                pq_name = PhysicalQuantityNames[i]
                axes = axes_list[i]
                colorbar = colorbar_list[i]
                axes.clear()
                img = axes.scatter(*[coor for _, coor in oneTDataDF[SpaceNames].items()], c=oneTDataDF[pq_name],
                                   cmap=plt.hot(), vmin=vminmax_d[pq_name][0], vmax=vminmax_d[pq_name][1],s=s)
                colorbar.update_normal(img)
                axes.set_title(f"{pq_name} ({t:.3f}s)")
                axes.set_xlabel('X')
                axes.set_ylabel('Y')

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True,exist_ok=True)
        fig_savepath = save_dir / f'{title}_plot.{save_type}'
        
        ani = animation.FuncAnimation(fig, func=update, frames=tqdm(selected_groups
            , desc=f"渲染动图中(时间帧:{time_frames},空间下采样:{down_k}倍)"), interval=50, init_func=init)

        # writer_classes = {
        #     'gif':animation.ImageMagickWriter,
        #     'mp4':animation.FFMpegWriter,
        #     'gif2':animation.PillowWriter,
        #     'html':animation.HTMLWriter,
        # }
        writer_strs = ('pillow','ffmpeg','ffmpeg_file','avconv','avconv_file','imagemagick','imagemagick_file','html')
        
        ani.save(fig_savepath, writer='ffmpeg', fps = 10)
        print(f"图片保存在{fig_savepath}")

    def plot_snapshot(self, t:float=0.0, save_dir='', title: str = None):
        if title is None:
            title = f'{self.name}_snapshot'

        t = self.DataDF["T"].iloc[(self.DataDF["T"]-t).abs().argsort().iloc[0]]
        snapDF = self.DataDF[self.DataDF["T"] == t]
        SpaceNames = list(self.SpaceNames)
        PhysicalQuantityNames = list(self.PhysicalQuantityNames)
        pq_num = len(self.PhysicalQuantityNames)
        pj = None if len(SpaceNames) < 3 else '3d'
        if pq_num == 0:
            fig = plt.figure(figsize=(FigSubSize, FigSubSize))
            fig.suptitle(title)
            axes = fig.add_subplot(1, 1, 1, projection=pj)
            img = axes.scatter(*[coor for _, coor in snapDF[SpaceNames].items()])
            axes.set_title(f"无物理量 ({t:.3f}s)")
        else:
            ncols = math.ceil(math.sqrt(pq_num))
            nrows = math.ceil(pq_num/ncols)
            fig = plt.figure(figsize=(FigSubSize*ncols, FigSubSize*nrows))
            fig.suptitle(title)

            for i in range(pq_num):
                pq_name = PhysicalQuantityNames[i]
                axes = fig.add_subplot(nrows, ncols, i+1, projection=pj)
                img = axes.scatter(*[coor for _, coor in snapDF[SpaceNames].items()],
                                c=snapDF[pq_name], cmap=plt.hot())
                fig.colorbar(img, ax=axes)
                axes.set_title(f"{pq_name} ({t:.3f}s)")
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True,exist_ok=True)
        fig_savepath = save_dir/ f'{title}_plot.jpg'
        print(f"图片保存在{fig_savepath}")
        plt.savefig(fig_savepath)

    @staticmethod
    def _plot_data(data4plot: np.ndarray, save_dir, title: str = ''):
        """已废弃

        Args:
            data4plot (np.ndarray): [description]
            save_dir ([type]): [description]
            title (str, optional): [description]. Defaults to ''.
        """
        fig = plt.figure(figsize=(FigSubSize, FigSubSize))
        xs = data4plot[:, -3]
        ys = data4plot[:, -2]
        ks = data4plot[:, -1]
        if data4plot.shape[-1] == 4:
            ts = data4plot[:, 0]
            ax = fig.add_subplot(111, projection='3d')
            img = ax.scatter(xs, ys, ts, c=ks, cmap=plt.hot())
        else:
            ax = fig.add_subplot(111)
            img = ax.scatter(xs, ys, c=ks, cmap=plt.hot())

        plt.title(title)
        fig.colorbar(img)
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True,exist_ok=True)
        fig_savepath = save_dir/ f'{title}_plot.png'
        print(f"图片保存在{fig_savepath}")
        plt.savefig(fig_savepath)
    
    def describe(self) -> dict:
        """描述自己的数据分布

        Returns:
            dict: 包含自己数据分布描述的字典
        """        
        desc = {
            '时空坐标点个数':self.DataDF.shape[0],
            '时间点个数':self.DataDF["T"].unique().shape[0],
            '时间T范围':(self.DataDF["T"].min(),self.DataDF["T"].max()),
            '空间维数':len(self.SpaceNames),
        }
        for sn in self.SpaceNames:
            desc[f'空间轴{sn}个数'] = self.DataDF[sn].unique().shape[0]
            desc[f'空间轴{sn}范围'] = (self.DataDF[sn].min(),self.DataDF[sn].max())
        
        for pqn in self.PhysicalQuantityNames:
            desc[f'物理量{pqn}范围'] = (self.DataDF[pqn].min(),self.DataDF[pqn].max())
            desc[f'物理量{pqn}均值'] = self.DataDF[pqn].mean()
            desc[f'物理量{pqn}标准差'] = self.DataDF[pqn].std()
        
        print(f"PhysicalProcessData - {self.name} 数据描述：")
        for k,v in desc.items():
            print(f"\t{k} : \t{v}")
        print()

        return desc

    def save(self,save_dir='',save_filename=None):
        """保存到文件

        Args:
            save_dir (str, optional): 保存到的文件夹路径. Defaults to ''.
            save_filename ([type], optional): 保存到文件名称. Defaults to None.
        """
        if save_filename is None: save_filename = self.name
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True,exist_ok=True)
        save_path = save_dir / save_filename + '.pkl.zip'

        self.DataDF.to_pickle(save_path,protocol=4)
    
    def load(self, save_path:str):
        """从保存好的文件中读取

        Args:
            save_path (str): 保存文件路径
        """
        self.load_DataDF(pd.read_pickle(save_path))

        return self
    
    def load_DataDF(self, data_df:pd.DataFrame):
        self.DataDF = data_df.copy()
        self._init_names()


def read_file(save_path:str,name = None):
    """直接从保存好的文件中创建新的 PhysicalProcessData 对象，这是一个静态函数

    Args:
        save_path (str): 保存文件路径

    Returns:
        PhysicalProcessData: PhysicalProcessData 对象
    """
    ppd = PhysicalProcessData(name = os.path.basename(save_path) if name is None else name)
    ppd.load(save_path)
    return ppd


def read_DataDF(data_df:pd.DataFrame, name:str):
    """直接从DataFrame创建新的 PhysicalProcessData 对象，这是一个静态函数

    Args:
        data_df (pd.DataFrame): 正确格式的DataFrame
        name (str): 要赋予 PhysicalProcessData 的名字

    Returns:
        PhysicalProcessData: PhysicalProcessData 对象
    """        
    ppd = PhysicalProcessData(name)
    ppd.load_DataDF(data_df)
    return ppd

def combine(*ppds:PhysicalProcessData,name:str = None):
    if name is None:
        name = f'combined_{ppds[0].name}...'
    return read_DataDF(pd.concat([ppd.DataDF for ppd in ppds]),name=name)

def get_my_data(name = None,pkl_data_fp='my_data/stokes_uv.pkl.zip'):
    if name is None: name = os.path.basename(pkl_data_fp)
    print(f"读取数据：{pkl_data_fp}")
    ppd = PhysicalProcessData.read_file(pkl_data_fp)
    return ppd

if __name__ == "__main__":
    """    ppd = getCylinder2D("Cylinder2D")
        # ppd.plot()
        # ppd.plot_snapshot()

        sub_pdd = ppd.subRangeData({"X":[-2,2],"Y":[-2,2]}).cutRangeData({"X":[-0.9,0.9],"Y":[-0.9,0.9]})
        sub_pdd.snapshot(t = 8.0).plot()
        sub_pdd.plot_snapshot()
        sub_pdd.plot()"""

    ppd = get_my_data()
    ppd.plot(save_type='gif')
    ppd.plot_snapshot()
