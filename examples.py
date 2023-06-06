import torch as th
from PhysicalProcessDataClass import PhysicalProcessData as PPD,combine

def create_TXY_by_range(T_range=(0, 10), X_range=(0, 1), Y_range=(0, 1), Tn=100, Xn=40, Yn=40):
    T_star, X_star, Y_star = th.meshgrid(th.linspace(T_range[0], T_range[1], Tn), th.linspace(
        X_range[0], X_range[1], Xn), th.linspace(Y_range[0], Y_range[1], Yn))
    return T_star.flatten(), X_star.flatten(), Y_star.flatten()

class Example_1:
    Info = 'On classical iterative subdomain methods for the Stokes/Darcy problem Exam 2'
    NS_area = dict(T_range=(0,0),X_range=(0, 1), Y_range=(1, 2), Tn=1, Xn=40, Yn=40)
    Darcy_area = dict(T_range=(0,0),X_range=(0, 1), Y_range=(0, 1), Tn=1, Xn=40, Yn=40)
    
    @staticmethod
    def NS_th(T: th.Tensor,X: th.Tensor,Y: th.Tensor):
        """输入时空坐标，输出对应的宏观物理量 U、V、P

        Args:
            TXY (th.Tensor): 时空坐标 (0,1) x (0,1)

        Returns:
            [type]: [description]
        """
        nu = 1.0
        g = 1.0
        kappa = 1.0

        U = Y**2 - 2*Y + 1
        V = X**2 - X
        P = 2 * nu * (X + Y - 1) + g/3*kappa

        return U,V,P
    
    @staticmethod
    def Darcy_th(T: th.Tensor,X: th.Tensor,Y: th.Tensor):
        """输入时空坐标，输出对应的Darcy的宏观物理量 U、V、P

        Args:
            TXY (th.Tensor): 时空坐标

        Returns:
            [type]: [description]
        """

        U = X*(Y - 1) - (1 - X)*(Y - 1) - 2
        V = -X*(1 - X) - Y**2 + 2*Y - 1
        P = X*(1 - X)*(Y - 1) + 2*X + Y**3/3 - Y**2 + Y

        return U,V,P
    
class Example_2:
    Info = '不可压缩流体流动及其耦 合问题的时空有限元算法_华东师范大学_2021 5.4.1'
    NS_area = dict(T_range=(0,1),X_range=(0, 1), Y_range=(1, 2), Tn=100, Xn=40, Yn=40)
    Darcy_area = dict(T_range=(0,1),X_range=(0, 1), Y_range=(0, 1), Tn=100, Xn=40, Yn=40)
    
    @staticmethod
    def NS_th(T: th.Tensor,X: th.Tensor,Y: th.Tensor):
        """输入时空坐标，输出对应的宏观物理量 U、V、P

        Args:
            TXY (th.Tensor): 时空坐标 (0,1) x (0,1)

        Returns:
            [type]: [description]
        """
        cos = th.cos 
        pi = th.pi 
        sin = th.sin 
        
        wf = 4
        
        U = (X**2*(Y - 1)**2 + Y)*cos(wf*T)
        V = -2*X*(Y - 1)**3*cos(wf*T)/3 + (-pi*sin(pi*X) + 2)*cos(T)
        P = (-pi*sin(pi*X) + 2)*sin(pi*Y/2)*cos(T)

        return U,V,P
    
    @staticmethod
    def Darcy_th(T: th.Tensor,X: th.Tensor,Y: th.Tensor):
        """输入时空坐标，输出对应的Darcy的宏观物理量 U、V、P

        Args:
            TXY (th.Tensor): 时空坐标

        Returns:
            [type]: [description]
        """
        cos = th.cos 
        pi = th.pi 
        sin = th.sin 
        
        U = pi**2*(-Y - cos(pi*Y) + 1)*cos(T)*cos(pi*X)
        V = -(-pi*sin(pi*X) + 2)*(pi*sin(pi*Y) - 1)*cos(T)
        P = (-pi*sin(pi*X) + 2)*(-Y - cos(pi*Y) + 1)*cos(T)

        return U,V,P

class Example_Li2018:
    Info = 'Li2018_Article_DiscontinuousFiniteVolumeEleme_2018 exam2'
    NS_area = dict(T_range=(0,1),X_range=(0, 1), Y_range=(1, 2), Tn=100, Xn=40, Yn=40)
    Darcy_area = dict(T_range=(0,1),X_range=(0, 1), Y_range=(0, 1), Tn=100, Xn=40, Yn=40)
    Interface_area = dict(T_range=(0,1),X_range=(0, 1), Y_range=(1, 1), Tn=100, Xn=400, Yn=1)
    
    @staticmethod
    def NS_loss_f(T: th.Tensor,X: th.Tensor,Y: th.Tensor):
        sin = th.sin 
        cos = th.cos 
        pi = th.pi
        f_1 = sin(T)*sin(pi*Y)*cos(pi*X) - 2*pi**2*sin(pi*Y)*cos(T)*cos(pi*X) + pi*cos(T)*cos(pi*X)
        f_2 = - sin(T)*sin(pi*X)*cos(pi*Y) + 2*pi**2*sin(pi*X)*cos(T)*cos(pi*Y)
        
        return f_1,f_2
    
    @staticmethod
    def NS_th(T: th.Tensor,X: th.Tensor,Y: th.Tensor):
        """输入时空坐标，输出对应的宏观物理量 U、V、P

        Args:
            TXY (th.Tensor): 时空坐标 (0,1) x (0,1)

        Returns:
            [type]: [description]
        """
        cos = th.cos 
        pi = th.pi 
        sin = th.sin 

        U = -sin(pi*Y)*cos(T)*cos(pi*X)
        V = sin(pi*X)*cos(T)*cos(pi*Y)
        P = sin(pi*X)*cos(T) + 0.0*Y

        return U,V,P
    
    @staticmethod
    def Darcy_loss_f(T: th.Tensor,X: th.Tensor,Y: th.Tensor):
        f1 = th.tensor([0.0,0.0])
        
        sin = th.sin 
        cos = th.cos 
        pi = th.pi
        f2 = - Y*sin(T)*sin(pi*X) + pi**2*Y*sin(pi*X)*cos(T)
        
        return f1,f2
    
    @staticmethod
    def Darcy_th(T: th.Tensor,X: th.Tensor,Y: th.Tensor):
        """输入时空坐标，输出对应的Darcy的宏观物理量 U、V、P

        Args:
            TXY (th.Tensor): 时空坐标

        Returns:
            [type]: [description]
        """
        cos = th.cos 
        pi = th.pi 
        sin = th.sin 
        
        U = -pi*Y*cos(T)*cos(pi*X)
        V = -sin(pi*X)*cos(T) + 0.0*Y
        P = Y*sin(pi*X)*cos(T)

        return U,V,P
    
class Example_Longtime2012:
    Info = '''Long time stability of four methods for splitting the evolutionary Stokes–Darcy problem into Stokes and Darcy subproblems_2012
    Omiga_f = (0,1) x (1,2)
    Omiga_p = (0,1) x (0,1)'''
    NS_area = dict(T_range=(0,1),X_range=(0, 1), Y_range=(1, 2), Tn=50, Xn=50, Yn=50)
    Darcy_area = dict(T_range=(0,1),X_range=(0, 1), Y_range=(0, 1), Tn=50, Xn=50, Yn=50)
    Interface_area = dict(T_range=(0,1),X_range=(0, 1), Y_range=(1, 1), Tn=50, Xn=500, Yn=1)
    
    @staticmethod
    def NS_input_f(T: th.Tensor,X: th.Tensor,Y: th.Tensor):
        sin = th.sin 
        cos = th.cos 
        pi = th.pi
        x = X
        y = Y
        t = T
        rho = 1
        f_1 = -rho*(-6*x*(y - 1)**2*(x**2*(y - 1)**2 + y)*cos(t)**2 + (2*x**2*(y - 1) + 1)*(2*x*(y - 1)**3 + 3*pi*sin(pi*x) - 6)*cos(t)**2 + 9*(x**2*(y - 1)**2 + y)*sin(t))/27 - 2*x**2*cos(t)/3 - 2*(y - 1)**2*cos(t)/3 - pi**2*sin(pi*y/2)*cos(t)*cos(pi*x)/3
        f_2 = rho*(2*x*(y - 1)**2*(2*x*(y - 1)**3 + 3*pi*sin(pi*x) - 6)*cos(t)**2 - (x**2*(y - 1)**2 + y)*(2*(y - 1)**3 + 3*pi**2*cos(pi*x))*cos(t)**2 + 3*(2*x*(y - 1)**3 + 3*pi*sin(pi*x) - 6)*sin(t))/27 + 4*x*(y - 1)*cos(t)/3 - pi*(pi*sin(pi*x) - 2)*cos(t)*cos(pi*y/2)/6 - pi**3*sin(pi*x)*cos(t)/3
        
        return f_1,f_2
    
    @staticmethod
    def NS_th(T: th.Tensor,X: th.Tensor,Y: th.Tensor):
        """输入时空坐标，输出对应的宏观物理量 U、V、P

        Args:
            TXY (th.Tensor): 时空坐标 (0,1) x (0,1)

        Returns:
            [type]: [description]
        """
        cos = th.cos 
        pi = th.pi 
        sin = th.sin 

        U = (X**2*(Y - 1)**2 + Y)*cos(T) / 3
        V = (-2*X*(Y - 1)**3/3 - pi*sin(pi*X) + 2)*cos(T) / 3
        P = (-pi*sin(pi*X) + 2)*sin(pi*Y/2)*cos(T) / 3

        return U,V,P
    
    @staticmethod
    def Darcy_input_f(T: th.Tensor,X: th.Tensor,Y: th.Tensor):
        f1 = th.tensor([0.0,0.0])
        
        sin = th.sin 
        cos = th.cos 
        pi = th.pi
        x = X
        y = Y
        t = T

        f2 = - (pi*sin(pi*x) - 2)*(y + cos(pi*y) - 1)*sin(t) / 3 + pi**2*(pi*sin(pi*x) - 2)*cos(t)*cos(pi*y)/3 + pi**3*(y + cos(pi*y) - 1)*sin(pi*x)*cos(t)/3
        
        return f1,f2
    
    @staticmethod
    def Darcy_th(T: th.Tensor,X: th.Tensor,Y: th.Tensor):
        """输入时空坐标，输出对应的Darcy的宏观物理量 U、V、P

        Args:
            TXY (th.Tensor): 时空坐标

        Returns:
            [type]: [description]
        """
        cos = th.cos 
        pi = th.pi 
        sin = th.sin 
        
        U = pi**2*(-Y - cos(pi*Y) + 1)*cos(T)*cos(pi*X) / 3
        V = -(-pi*sin(pi*X) + 2)*(pi*sin(pi*Y) - 1)*cos(T) / 3
        P = (-pi*sin(pi*X) + 2)*(-Y - cos(pi*Y) + 1)*cos(T) / 3

        return U,V,P

class Example_Longtime2012_stable:
    Info = '''Long time stability of four methods for splitting the evolutionary Stokes–Darcy problem into Stokes and Darcy subproblems_2012
    Omiga_f = (0,1) x (1,2)
    Omiga_p = (0,1) x (0,1)'''
    NS_area = dict(T_range=(0,0),X_range=(0, 1), Y_range=(1, 2), Tn=1, Xn=40, Yn=40)
    Darcy_area = dict(T_range=(0,0),X_range=(0, 1), Y_range=(0, 1), Tn=1, Xn=40, Yn=40)
    Interface_area = dict(T_range=(0,0),X_range=(0, 1), Y_range=(1, 1), Tn=1, Xn=400, Yn=1)
    
    @staticmethod
    def NS_input_f(T: th.Tensor,X: th.Tensor,Y: th.Tensor):
        sin = th.sin 
        cos = th.cos 
        pi = th.pi
        x = X
        y = Y
        t = T
        rho = 1
        f_1 = rho*(6*x*(y - 1)**2*(x**2*(y - 1)**2 + y) - (x**2*(2*y - 2) + 1)*(2*x*(y - 1)**3 + 3*pi*sin(pi*x) - 6))/27 - 2*x**2/3 - 2*(y - 1)**2/3 - pi**2*sin(pi*y/2)*cos(pi*x)/3
        f_2 = rho*(2*x*(y - 1)**2*(2*x*(y - 1)**3 + 3*pi*sin(pi*x) - 6) - (x**2*(y - 1)**2 + y)*(2*(y - 1)**3 + 3*pi**2*cos(pi*x)))/27 + 4*x*(y - 1)/3 - pi*(pi*sin(pi*x) - 2)*cos(pi*y/2)/6 - pi**3*sin(pi*x)/3
        
        return f_1,f_2
    
    @staticmethod
    def NS_th(T: th.Tensor,X: th.Tensor,Y: th.Tensor):
        """输入时空坐标，输出对应的宏观物理量 U、V、P

        Args:
            TXY (th.Tensor): 时空坐标 (0,1) x (0,1)

        Returns:
            [type]: [description]
        """
        # cos = th.cos 
        pi = th.pi 
        sin = th.sin 
        x = X
        y = Y

        U = x**2*(y - 1)**2/3 + y/3
        V = -2*x*(y - 1)**3/9 - pi*sin(pi*x)/3 + 2/3
        P = (-pi*sin(pi*x) + 2)*sin(pi*y/2)/3

        return U,V,P
    
    @staticmethod
    def Darcy_input_f(T: th.Tensor,X: th.Tensor,Y: th.Tensor):
        f1 = th.tensor([0.0,0.0])
        
        sin = th.sin 
        cos = th.cos 
        pi = th.pi
        x = X
        y = Y
        # t = T

        f2 = pi**2*(pi*sin(pi*x) - 2)*cos(pi*y)/3 + pi**3*(y + cos(pi*y) - 1)*sin(pi*x)/3
        
        return f1,f2
    
    @staticmethod
    def Darcy_th(T: th.Tensor,X: th.Tensor,Y: th.Tensor):
        """输入时空坐标，输出对应的Darcy的宏观物理量 U、V、P

        Args:
            TXY (th.Tensor): 时空坐标

        Returns:
            [type]: [description]
        """
        cos = th.cos 
        pi = th.pi 
        sin = th.sin 
        
        x = X 
        y = Y 
        
        U = pi**2*(-y - cos(pi*y) + 1)*cos(pi*x)/3
        V = (pi*sin(pi*x) - 2)*(pi*sin(pi*y) - 1)/3
        P = (pi*sin(pi*x) - 2)*(y + cos(pi*y) - 1)/3

        return U,V,P

class Example_Decoupled2009:
    Info = '''DECOUPLED SCHEMES FOR A NON-STATIONARY MIXED STOKES-DARCY MODEL-已解锁 案例2
    Omiga_f = (0,1) x (1,2)
    Omiga_p = (0,1) x (0,1)'''
    NS_area = dict(T_range=(0,1),X_range=(0, 1), Y_range=(1, 2), Tn=50, Xn=40, Yn=40)
    Darcy_area = dict(T_range=(0,1),X_range=(0, 1), Y_range=(0, 1), Tn=50, Xn=40, Yn=40)
    Interface_area = dict(T_range=(0,1),X_range=(0, 1), Y_range=(1, 1), Tn=50, Xn=40, Yn=1)
    
    @staticmethod
    def NS_input_f(T: th.Tensor,X: th.Tensor,Y: th.Tensor):
        sin = th.sin 
        cos = th.cos 
        pi = th.pi
        x = X
        y = Y
        t = T
        rho = 1
        f_1 = -(y**2 - 2*y + 1)*sin(t)
        f_2 = -(x**2 - x)*sin(t)
        
        return f_1,f_2
    
    @staticmethod
    def NS_th(T: th.Tensor,X: th.Tensor,Y: th.Tensor, nu = 1.0, kappa = 1.0):
        """输入时空坐标，输出对应的宏观物理量 U、V、P

        Args:
            TXY (th.Tensor): 时空坐标 (0,1) x (0,1)

        Returns:
            [type]: [description]
        """
        cos = th.cos 
        pi = th.pi 
        sin = th.sin 
        x = X
        y = Y
        t = T 
        
        U = (y**2 - 2*y + 1) * cos(t) + 0.0 * x
        V = (x**2 - x) * cos(t) + 0.0 * y
        P = (2 * nu * (x + y - 1) + 1/(3*kappa)) * cos(t)

        return U,V,P
    
    @staticmethod
    def Darcy_input_f(T: th.Tensor,X: th.Tensor,Y: th.Tensor, nu = 1.0,kappa = 1.0):
        f1 = th.tensor([0.0,0.0])
        
        sin = th.sin 
        cos = th.cos 
        pi = th.pi
        x = X
        y = Y
        t = T

        f2 = (-f2*kappa + (-2.0*kappa*nu*x + 1.0*x*(x - 1)*(y - 1) - y**3 / 3 + 1.0*y**2 - 1.0*y)*sin(t))/kappa
        
        return f1,f2
    
    @staticmethod
    def Darcy_th(T: th.Tensor,X: th.Tensor,Y: th.Tensor, nu = 1.0,kappa = 1.0):
        """输入时空坐标，输出对应的Darcy的宏观物理量 U、V、P

        Args:
            TXY (th.Tensor): 时空坐标

        Returns:
            [type]: [description]
        """
        cos = th.cos 
        pi = th.pi 
        sin = th.sin 
        x = X
        y = Y
        t = T
        
        U = -kappa*(2*nu + (-x*(y - 1) + (1 - x)*(y - 1))/kappa)*cos(t)
        V = -(x*(1 - x) + y**2 - 2*y + 1)*cos(t)
        P = (1/kappa * (x*(1-x)*(y-1) + y**3/3 - y**2 + y) + 2 *nu * x) * cos(t)

        return U,V,P

class Example_Qingdao2016_1:
    Info = '''基于有限元方法求解的具有交界面边界条件的Stokes_Darcy耦合系统_青岛大学_2016 算例1
    Omiga_s = (0,1) x (-0.25,0)
    Omiga_p = (0,1) x (0,0.75)
    稳态'''
    NS_area = dict(T_range=(0,0),X_range=(0,1), Y_range=(-0.25,0), Tn=1, Xn=40, Yn=10)
    Darcy_area = dict(T_range=(0,0),X_range=(0, 1), Y_range=(0, 0.75), Tn=1, Xn=40, Yn=30)
    Interface_area = dict(T_range=(0,0),X_range=(0, 1), Y_range=(0, 0), Tn=1, Xn=400, Yn=1)
    
    @staticmethod
    def NS_input_f(T: th.Tensor,X: th.Tensor,Y: th.Tensor):
        sin = th.sin 
        cos = th.cos 
        exp = th.exp
        pi = th.pi
        x = X
        y = Y
        # t = T
        rho = 1
        f_1 = (rho*(6*x*y**2*(x**2*y**2*exp(y) + 1) + (-2*x**2*y*exp(y) + 1)*(2*x*y**3 + 3*pi*sin(pi*x) - 6)) + 3*(-2*x**2 - 2*y**2 + pi**2*cos(pi*x)*cos(2*pi*y))*exp(y) - 3)*exp(-y)/3
        f_2 = (rho*(2*x*y**2*(2*x*y**3 + 3*pi*sin(pi*x) - 6)*exp(y) - (2*y**3 + 3*pi**2*cos(pi*x))*(x**2*y**2*exp(y) + 1)) + 3*(4*x*y - 2*pi*(pi*sin(pi*x) - 2)*sin(2*pi*y) - pi**3*sin(pi*x))*exp(y))*exp(-y)/3

        return f_1,f_2
    
    @staticmethod
    def NS_th(T: th.Tensor,X: th.Tensor,Y: th.Tensor):
        """输入时空坐标，输出对应的宏观物理量 U、V、P

        Args:
            TXY (th.Tensor): 时空坐标 (0,1) x (0,1)

        Returns:
            [type]: [description]
        """
        cos = th.cos 
        pi = th.pi 
        sin = th.sin 
        exp = th.exp
        x = X
        y = Y

        U = x**2*y**2 + exp(-y)
        V = -2*x*y**3/3 - pi*sin(pi*x) + 2
        P = (pi*sin(pi*x) - 2)*cos(2*pi*y)

        return U,V,P
    
    @staticmethod
    def Darcy_input_f(T: th.Tensor,X: th.Tensor,Y: th.Tensor):
        f1 = th.tensor([0.0,0.0])
        
        sin = th.sin 
        cos = th.cos 
        pi = th.pi
        x = X
        y = Y
        # t = T

        f2 = pi**3*(y + cos(pi*y))*sin(pi*x) + pi**2*(pi*sin(pi*x) - 2)*cos(pi*y)
        
        return f1,f2
    
    @staticmethod
    def Darcy_th(T: th.Tensor,X: th.Tensor,Y: th.Tensor):
        """输入时空坐标，输出对应的Darcy的宏观物理量 U、V、P

        Args:
            TXY (th.Tensor): 时空坐标

        Returns:
            [type]: [description]
        """
        cos = th.cos 
        pi = th.pi 
        sin = th.sin 
        
        x = X
        y = Y
        
        U = -pi**2*(y + cos(pi*y))*cos(pi*x)
        V = (pi*sin(pi*x) - 2)*(pi*sin(pi*y) - 1)
        P = (y + cos(pi*y))*(pi*sin(pi*x) - 2)

        return U,V,P

class Example_SIAM2013:
    Info = '''1--SIAM2013 算例1
    Omiga_s = (0,1) x (1,2)
    Omiga_p = (0,1) x (0,1)
    稳态'''
    NS_area = dict(T_range=(0,1),X_range=(0, 1), Y_range=(1, 2), Tn=50, Xn=50, Yn=50)
    Darcy_area = dict(T_range=(0,1),X_range=(0, 1), Y_range=(0, 1), Tn=50, Xn=50, Yn=50)
    Interface_area = dict(T_range=(0,1),X_range=(0, 1), Y_range=(1, 1), Tn=50, Xn=500, Yn=1)
    
    @staticmethod
    def NS_input_f(T: th.Tensor,X: th.Tensor,Y: th.Tensor,kappa=1.0,a=0.0):
        sin = th.sin 
        cos = th.cos 
        pi = th.pi
        x = X
        y = Y
        t = T
        rho = 1
        
        f_1 = (2*x - 1)*(y - 1)*sin(t) - (2*x - 1)*(x*(x - 1) + (y - 1)**2)*cos(t)**2 + (4*x - 2)*(y - 1)**2*cos(t)**2 - (2*x - 1)*(y - 1)*cos(t)/kappa
        f_2 = -4*cos(t) - (2*x - 1)**2*(y - 1)*cos(t)**2 + 2*(y - 1)*(x*(x - 1) + (y - 1)**2)*cos(t)**2 - (x*(x - 1) + (y - 1)**2)*sin(t) + (-x*(x - 1) + y**2 - 2*y + 1)*cos(t)/kappa
        
        return f_1,f_2
    
    @staticmethod
    def NS_th(T: th.Tensor,X: th.Tensor,Y: th.Tensor, kappa = 1.0, a = 0.0):
        """输入时空坐标，输出对应的宏观物理量 U、V、P

        Args:
            TXY (th.Tensor): 时空坐标 (0,1) x (0,1)

        Returns:
            [type]: [description]
        """
        cos = th.cos 
        pi = th.pi 
        sin = th.sin 
        x = X
        y = Y
        t = T 
        
        
        U = ((1 - 2*x) * (y - 1)) * cos(t)
        V = (x * (x - 1) + (y - 1)**2) * cos(t)
        P = 1/kappa * (x*(1-x)*(y-1) + y**3/3 - y**2 + y - a ) * cos(t)

        return U,V,P
    
    @staticmethod
    def Darcy_input_f(T: th.Tensor,X: th.Tensor,Y: th.Tensor,kappa=1.0,a=0.0):
        f1 = th.tensor([0.0,0.0])
        
        sin = th.sin 
        cos = th.cos 
        pi = th.pi
        x = X
        y = Y
        t = T

        f2 = (1.0*a + 1.0*x*(x - 1)*(y - 1) - 0.333333333333333*y**3 + 1.0*y**2 - 1.0*y)*sin(t)/kappa
        
        return f1,f2
    
    @staticmethod
    def Darcy_th(T: th.Tensor,X: th.Tensor,Y: th.Tensor,kappa = 1.0,a = 0.0):
        """输入时空坐标，输出对应的Darcy的宏观物理量 U、V、P

        Args:
            TXY (th.Tensor): 时空坐标

        Returns:
            [type]: [description]
        """
        cos = th.cos 
        pi = th.pi 
        sin = th.sin 
        x = X
        y = Y
        t = T
        
        
        U = -(-x*(y - 1) + (1 - x)*(y - 1))*cos(t)
        V = -(x*(1 - x) + y**2 - 2*y + 1)*cos(t)
        P = 1/kappa * (x*(1-x)*(y-1) + y**3/3 - y**2 + y - a) * cos(t)

        return U,V,P
    
    
    
def create_ppd(name,NS_area,Darcy_area,NS_th,Darcy_th):
    Ts,Xs,Ys = create_TXY_by_range(**NS_area)
    Td,Xd,Yd = create_TXY_by_range(**Darcy_area)
    Us,Vs,Ps = NS_th(Ts,Xs,Ys)
    Ud,Vd,Pd = Darcy_th(Td,Xd,Yd)
    arg_ns = ('T','X','Y','U','V','P')
    stokes_ppd = PPD(name=name + '-Stokes',**{n:x.detach().cpu().numpy() for n,x in zip(arg_ns,(Ts,Xs,Ys,Us,Vs,Ps))})
    darcy_ppd = PPD(name=name + '-Darcy',**{n:x.detach().cpu().numpy() for n,x in zip(arg_ns,(Td,Xd,Yd,Ud,Vd,Pd))})
    return stokes_ppd,darcy_ppd
    
def example2ppd(exam:Example_1):
    return create_ppd(exam.__name__,exam.NS_area,exam.Darcy_area,exam.NS_th,exam.Darcy_th)


def plot_example(exam:Example_1):
    ppd = combine(*example2ppd(exam))
    # ppd.plot_snapshot(save_dir='images')
    ppd.plot(save_dir='images')
    print(ppd.describe())

if __name__ == '__main__':
    plot_example(Example_Decoupled2009)