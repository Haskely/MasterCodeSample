
from typing import Iterable
from sympy import *
import sympy
init_printing()


def nable(ys, xs):
    if not (isinstance(ys, Iterable) or hasattr(ys, '__getitem__')):
        ys = [ys]
    elif not (isinstance(xs, Iterable) or hasattr(xs, '__getitem__')):
        xs = [xs]
    return Matrix([[diff(y, x) for y in ys] for x in xs])


def laplace(ys, xs):
    return Matrix([sum([diff(y, x, 2) for x in xs]) for y in ys])


def diver(ys, xs):
    return sum([diff(y, x) for y, x in zip(ys, xs)])


def U_dot_nable(us, ys, xs):
    return Matrix([sum([u*diff(y, x) for u, x in zip(us, xs)]) for y in ys])


def DD_(ys, xs):
    nable_Y = nable(ys, xs)
    DD_Y = (nable_Y + nable_Y.T) / 2
    return DD_Y


def TT_(us, ps, xs, nu=Symbol('nu')):
    d = len(us)
    return 2 * nu * DD_(us, xs) - ps * eye(d)


t, x, y, rho, nu, alpha, beta, kappa, S0 = symbols(
    't,x,y,rho,nu,alpha,beta,kappa,S_0')
Us = Matrix([Function('Us1')(t, x, y), Function('Us2')(t, x, y)])
Ps = Function('Ps')(t, x, y)
Ud = Matrix([Function('Ud1')(t, x, y), Function('Ud2')(t, x, y)])
Pd = Function('Pd')(t, x, y)
f = Matrix([Function('f_1')(t, x, y), Function('f_2')(t, x, y)])
f1 = Matrix([Function('f1_1')(t, x, y), Function('f1_2')(t, x, y)])

def Navier_Stokes_PDE(Us: Matrix = Us,
                      Ps: Mul = Ps,
                      TimeVar: Symbol = t,
                      SpaceVars: list[Symbol] = [x, y],
                      rho=rho,
                      nu=nu,
                      f=f,
                      is_nonliner_type = True
                      ):
    
    if is_nonliner_type:
        Eqn1 = rho * (diff(Us, TimeVar) + U_dot_nable(Us, Us, SpaceVars)) - \
            nu * laplace(Us, SpaceVars) + nable(Ps, SpaceVars) - f
    else:
        Eqn1 = rho * (diff(Us, TimeVar)) - nu * laplace(Us,
                                                    SpaceVars) + nable(Ps, SpaceVars) - f
    Eqn2 = diver(Us, SpaceVars)

    return simplify(Matrix((Eqn1, Eqn2)))


def Darcy_PDE(Ud: Matrix = Ud,
              Pd: Mul = Pd,
              TimeVar: Symbol = Symbol('t'),
              SpaceVars: list[Symbol] = symbols('x,y'),
              nu=nu,
              kappa=kappa,
              rho=rho,
              S0=S0,
              f2=Symbol('f2'),
              is_nonliner_type = False):

    Eqn1 = Ud + kappa * nable(Pd, SpaceVars)
    Eqn2 = S0 * diff(Pd, TimeVar) + diver(Ud, SpaceVars) - f2

    return simplify(Matrix((Eqn1, Eqn2)))


def Interface(
    Us: Matrix = Us,
    Ps: Mul = Ps,
    Ud: Matrix = Ud,
    Pd: Mul = Pd,
    TimeVar: Symbol = t,
    SpaceVars: list[Symbol] = [x, y],
    Ns=Matrix(symbols('Ns_1,Ns_2')),  # Stokes/Darcy 边界法向量，方向指向Darcy
    Tau=Matrix(symbols('Tau_1,Tau_2')),  # Stokes/Darcy 边界切向量
    BVar=y,
    BVarValue=1,
    nu=nu,
    rhog=Symbol('rho g'),
    alpha=alpha,
    kappa=kappa,

    is_T_type=False,
    is_care_Ud=True
):

    # d = len(SpaceVars)
    # nu * alpha * sqrt(d)/kappa

    nable_Us = nable(Us, SpaceVars)
    if is_T_type:
        D_Us = (nable_Us + nable_Us.T) / 2
        Temp = 2 * nu * D_Us
    else:
        Temp = nable_Us

    if is_care_Ud:
        Temp2 = (Us - Ud)
    else:
        Temp2 = Us

    Eqn1 = (Us.dot(Ns) - Ud.dot(Ns)).subs(BVar, BVarValue)
    Eqn2 = (Ps - nu*Ns.dot(Temp * Ns) - rhog * Pd).subs(BVar, BVarValue)
    Eqn3 = (-nu*Tau.dot(Temp * Ns) - alpha * sqrt(nu/kappa) * Temp2.dot(Tau)).subs(BVar, BVarValue)

    return simplify(Matrix((Eqn1, Eqn2, Eqn3)))


def vertify(Us, Ps, Ud, Pd,
            Ns=Matrix([0, -1]),
            Tau=Matrix([1, 0]),
            BVar=y,
            BVarValue=1,
            is_ns_nonliner = False,
            is_darcy_nonliner = False,
            is_T_type=False,
            is_care_Ud=False,
            rho = 1,
            nu = 1,
            kappa = 1):
    NS_eqns = Navier_Stokes_PDE(Us, Ps, 
                                rho=rho, 
                                nu=nu,
                                is_nonliner_type=is_ns_nonliner)
    Darcy_eqns = Darcy_PDE(Ud, Pd, 
                           kappa=kappa, 
                           nu=nu, 
                           rho=rho, 
                           S0=1.0,
                           is_nonliner_type=is_darcy_nonliner)
    Interface_eqns = Interface(
        Us,
        Ps,
        Ud,
        Pd,
        Ns=Ns,
        Tau=Tau,
        BVar=BVar,
        BVarValue=BVarValue,
        rhog=1,
        nu=nu,
        alpha=1,
        kappa = kappa,
        is_T_type=is_T_type,
        is_care_Ud=is_care_Ud)
    return dict(NS_eqns=NS_eqns, Darcy_eqns=Darcy_eqns, Interface_eqns=Interface_eqns)


def exam_Li2018():
    """Li2018_Article_DiscontinuousFiniteVolumeEleme
    
    Navier_Stokes_PDE liner
    Darcy_PDE liner
    Interface T_type not_care_Ud
    """
    a1,a2,a3 = symbols("a1,a2,a3")
    
    Us=Matrix([
        -sin(pi*y)*cos(pi*x)*cos(t),
        sin(pi*x)*cos(pi*y)*cos(t)
    ])
    Ps= sin(pi*x) * cos(t)
    Pd = (y) * sin(pi * x) * cos(t) 
    Ud=-nable(Pd, [x, y])
    
    Ns=Matrix([0, -1])
    Tau=Matrix([1, 0])
    BVar=y
    BVarValue=1
    
    is_ns_nonliner = True
    is_darcy_nonliner = False
    is_T_type=True
    is_care_Ud=False
    
    return dict(
        Us=Us,
        Ps=Ps,
        Pd=Pd,
        Ud=Ud,
        Ns=Ns,
        Tau=Tau,
        BVar=BVar,
        BVarValue=BVarValue,
        is_ns_nonliner = is_ns_nonliner,
        is_darcy_nonliner = is_darcy_nonliner,
        is_T_type = is_T_type,
        is_care_Ud = is_care_Ud,
    )
    


def exam_Longtime2012():
    """Long time stability of four methods for splitting the evolutionary Stokes–Darcy problem into Stokes and Darcy subproblems_2012
    Omiga_s = (0,1) x (1,2)
    Omiga_p = (0,1) x (0,1)
    """
    Us=Matrix([
            Rational(1, 3) * (x**2*(y-1)**2 + y) * cos(t),
            Rational(1, 3) * (-Rational(2, 3)*x*(y-1)**3 + 2 - pi * sin(pi*x)) * cos(t)
        ])
    Ps= Rational(1, 3) * (2 - pi * sin(pi*x)) * sin(pi * y/2) * cos(t)
    Pd = Rational(1, 3) * (2 - pi * sin(pi * x)) * (1 - y - cos(pi * y)) * cos(t)
    Ud=-nable(Pd, [x, y])
    
    Ns=Matrix([0, -1])
    Tau=Matrix([1, 0])
    BVar=y
    BVarValue=1
    
    is_ns_nonliner = True
    is_darcy_nonliner = False
    is_T_type=True
    is_care_Ud=True
    
    return dict(
        Us=Us,
        Ps=Ps,
        Pd=Pd,
        Ud=Ud,
        Ns=Ns,
        Tau=Tau,
        BVar=BVar,
        BVarValue=BVarValue,
        is_ns_nonliner = is_ns_nonliner,
        is_darcy_nonliner = is_darcy_nonliner,
        is_T_type = is_T_type,
        is_care_Ud = is_care_Ud,
    )

def exam_Longtime2012_stable():
    """Long time stability of four methods for splitting the evolutionary Stokes–Darcy problem into Stokes and Darcy subproblems_2012
    Omiga_s = (0,1) x (1,2)
    Omiga_p = (0,1) x (0,1)
    稳态
    """
    Us=Matrix([
            Rational(1, 3) * (x**2*(y-1)**2 + y),
            Rational(1, 3) * (-Rational(2, 3)*x*(y-1)**3 + 2 - pi * sin(pi*x))
        ])
    Ps= Rational(1, 3) * (2 - pi * sin(pi*x)) * sin(pi * y/2)
    Pd = Rational(1, 3) * (2 - pi * sin(pi * x)) * (1 - y - cos(pi * y))
    Ud=-nable(Pd, [x, y])
    
    Ns=Matrix([0, -1])
    Tau=Matrix([1, 0])
    BVar=y
    BVarValue=1
    
    is_ns_nonliner = True
    is_darcy_nonliner = False
    is_T_type=True
    is_care_Ud=True
    
    return dict(
        Us=Us,
        Ps=Ps,
        Pd=Pd,
        Ud=Ud,
        Ns=Ns,
        Tau=Tau,
        BVar=BVar,
        BVarValue=BVarValue,
        is_ns_nonliner = is_ns_nonliner,
        is_darcy_nonliner = is_darcy_nonliner,
        is_T_type = is_T_type,
        is_care_Ud = is_care_Ud,
    )
    
def exam_Domain2020_1():
    """Domain decomposition method for the fully-mixed Stokes–Darcy coupled problem exam 1
    Omiga_s = (0,1) x (1,2)
    Omiga_p = (0,1) x (0,1)
    这篇论文说他满足的是BJS条件。我要验证一下
    """
    Us=Matrix([
            Rational(1, 3) * (x**2*(y-1)**2 + y),
            Rational(1, 3) * (-Rational(2, 3)*x*(y-1)**3 + 2 - pi * sin(pi*x))
        ])
    Ps= Rational(1, 3) * (2 - pi * sin(pi*x)) * sin(pi * y/2)
    Pd = Rational(1, 3) * (2 - pi * sin(pi * x)) * (1 - y - cos(pi * y)) 
    Ud=-nable(Pd, [x, y])
    
    Ns=Matrix([0, -1])
    Tau=Matrix([1, 0])
    BVar=y
    BVarValue=1
    
    is_ns_nonliner = True
    is_darcy_nonliner = False
    is_T_type=True
    is_care_Ud=True
    
    return dict(
        Us=Us,
        Ps=Ps,
        Pd=Pd,
        Ud=Ud,
        Ns=Ns,
        Tau=Tau,
        BVar=BVar,
        BVarValue=BVarValue,
        is_ns_nonliner = is_ns_nonliner,
        is_darcy_nonliner = is_darcy_nonliner,
        is_T_type = is_T_type,
        is_care_Ud = is_care_Ud,
    )

def exam_Domain2020_2():
    """Domain decomposition method for the fully-mixed Stokes–Darcy coupled problem exam 3
    Omiga_s = (0,1) x (1,2)
    Omiga_p = (0,1) x (0,1)
    这篇论文说他满足的是BJS条件。我要验证一下
    """
    
    
    Us=Matrix([
            (y**2 - 2*y + 1) * cos(t),
            (x**2 - x) * cos(t)
        ])
    Ps= (2 *nu* (x + y - 1) + 1/(3*kappa)) * cos(t)
    Pd = (1/kappa * (x*(1-x)*(y-1) + y**3/3 - y**2 + y) + 2 *nu * x) * cos(t)
    Ud=-nable(Pd, [x, y])
    
    Ns=Matrix([0, -1])
    Tau=Matrix([1, 0])
    BVar=y
    BVarValue=1
    
    is_ns_nonliner = True
    is_darcy_nonliner = False
    is_T_type=True
    is_care_Ud=False
    
    return dict(
        Us=Us,
        Ps=Ps,
        Pd=Pd,
        Ud=Ud,
        Ns=Ns,
        Tau=Tau,
        BVar=BVar,
        BVarValue=BVarValue,
        is_ns_nonliner = is_ns_nonliner,
        is_darcy_nonliner = is_darcy_nonliner,
        is_T_type = is_T_type,
        is_care_Ud = is_care_Ud,
    )

def exam_ZheJiang2014_1():
    """Navier_Stokes_Darcy多区域耦合问题的多重网格方法_浙江大学_2014 算例2.1
    Omiga_s = (0,1) x (1,2)
    Omiga_p = (0,1) x (0,1)
    稳态
    """
    a1,a2,a3 = symbols("a1,a2,a3")
    
    Us=Matrix([
            a1 * (y**2 - 2 * y + 2 * x),
            a1 * (x**2 - x - 2*y)
        ])
    Ps= a1 * (-x**2*y + x*y + y**2 - 4)
    Pd = a1 * (-x**2*y*a2 + x*y + y**2)
    Ud=-nable(Pd, [x, y])
    
    Ns=Matrix([0, -1])
    Tau=Matrix([1, 0])
    BVar=y
    BVarValue=1
    
    is_ns_nonliner = True
    is_darcy_nonliner = False
    is_T_type=True
    is_care_Ud=True
    
    return dict(
        Us=Us,
        Ps=Ps,
        Pd=Pd,
        Ud=Ud,
        Ns=Ns,
        Tau=Tau,
        BVar=BVar,
        BVarValue=BVarValue,
        is_ns_nonliner = is_ns_nonliner,
        is_darcy_nonliner = is_darcy_nonliner,
        is_T_type = is_T_type,
        is_care_Ud = is_care_Ud,
    )

def exam_ZheJiang2014_2():
    """Navier_Stokes_Darcy多区域耦合问题的多重网格方法_浙江大学_2014 算例2.2
    Omiga_s = (0,1) x (1,2)
    Omiga_p = (0,1) x (0,1)
    稳态
    """
    Us=Matrix([
            Rational(1, 1) * (sin(pi*x)*sin(pi*y)),
            Rational(1, 1) * (cos(pi*x)*cos(pi*y)),
        ])
    Ps= Rational(1, 1) * (y**5 * cos(pi*x))
    Pd = Rational(1, 1) * (y * cos(pi*x))
    Ud=-nable(Pd, [x, y]) 
    
    Ns=Matrix([0, -1])
    Tau=Matrix([1, 0])
    BVar=y
    BVarValue=1
    
    is_ns_nonliner = True
    is_darcy_nonliner = False
    is_T_type=True
    is_care_Ud=False
    
    return dict(
        Us=Us,
        Ps=Ps,
        Pd=Pd,
        Ud=Ud,
        Ns=Ns,
        Tau=Tau,
        BVar=BVar,
        BVarValue=BVarValue,
        is_ns_nonliner = is_ns_nonliner,
        is_darcy_nonliner = is_darcy_nonliner,
        is_T_type = is_T_type,
        is_care_Ud = is_care_Ud,
    )

def exam_QingDao2016_1():
    """基于有限元方法求解的具有交界面边界条件的Stokes_Darcy耦合系统_青岛大学_2016 算例1
    Omiga_s = (0,1) x (-0.25,0)
    Omiga_p = (0,1) x (0,0.75)
    稳态
    """
    Us=Matrix([
            Rational(1, 1) * (x**2*y**2 + exp(-y)),
            Rational(1, 1) * (-Rational(2, 3)*x*y**3 + 2 - pi * sin(pi*x))
        ])
    Ps= Rational(1, 1) * -(2 - pi * sin(pi*x)) * cos(2 * pi * y )
    Pd = Rational(1, 1) * (2 - pi * sin(pi * x)) * (- y + cos(pi * (1-y))) 
    Ud=-nable(Pd, [x, y])
    
    Ns=Matrix([0, 1])
    Tau=Matrix([1, 0])
    BVar=y
    BVarValue=0
    
    is_ns_nonliner = True
    is_darcy_nonliner = False
    is_T_type=False
    is_care_Ud=True
    
    return dict(
        Us=Us,
        Ps=Ps,
        Pd=Pd,
        Ud=Ud,
        Ns=Ns,
        Tau=Tau,
        BVar=BVar,
        BVarValue=BVarValue,
        is_ns_nonliner = is_ns_nonliner,
        is_darcy_nonliner = is_darcy_nonliner,
        is_T_type = is_T_type,
        is_care_Ud = is_care_Ud,
    )

def exam_QingDao2016_2():
    """基于有限元方法求解的具有交界面边界条件的Stokes_Darcy耦合系统_青岛大学_2016 算例1
    Omiga_s = (0,1) x (-0.25,0)
    Omiga_p = (0,1) x (0,0.75)
    非稳态
    """
    Us=Matrix([
            Rational(1, 1) * (x**2*y**2 + exp(-y)) * cos(2*pi*t),
            Rational(1, 1) * (-Rational(2, 3)*x*y**3 + 2 - pi * sin(pi*x)) * cos(2*pi*t)
        ])
    Ps= Rational(1, 1) * -(2 - pi * sin(pi*x)) * cos(2 * pi * y ) * cos(2*pi*t)
    Pd = Rational(1, 1) * (2 - pi * sin(pi * x)) * (- y + cos(pi * (1-y)))  * cos(2*pi*t)
    Ud=-nable(Pd, [x, y])
    
    Ns=Matrix([0, 1])
    Tau=Matrix([1, 0])
    BVar=y
    BVarValue=0
    
    is_ns_nonliner = True
    is_darcy_nonliner = False
    is_T_type=True
    is_care_Ud=True
    
    return dict(
        Us=Us,
        Ps=Ps,
        Pd=Pd,
        Ud=Ud,
        Ns=Ns,
        Tau=Tau,
        BVar=BVar,
        BVarValue=BVarValue,
        is_ns_nonliner = is_ns_nonliner,
        is_darcy_nonliner = is_darcy_nonliner,
        is_T_type = is_T_type,
        is_care_Ud = is_care_Ud,
    )

def tolatex(exam):
    for k,v in exam.items():
        v = simplify(v)
        print(f'{k}:\n \t{latex(v)}')

def _main(exam):
    vertify_res = vertify(**exam)
    
    for k,v in exam.items():
        v = simplify(v)
        print(f'{k}:\n \t{v}')
        
    for k,v in vertify_res.items():
        v = simplify(v)
        print(f'{k}:\n \t{v}')
    
if __name__ == '__main__':
    exam = exam_Li2018()
    _main(exam)
    # tolatex(exam)