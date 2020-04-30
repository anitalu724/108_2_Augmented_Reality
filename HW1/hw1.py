import numpy as np
import sys, cv2, time
import math
from math import *
from sympy import *
from termcolor import colored

F = 4.73    # unit = mm
Epsilon = 10**(-8)

# from matplotlib import pyplot as plt
def length(a, b):
    if len(a)!= len(b):
        print("Length computing error!\n")
        return False
    else:
        sum = 0
        for i in range(len(a)):
            sum += (a[i]-b[i])**2
        return sum**0.5             # return unit:"mm"  

def find_S(_s, real, pic):
    S = _s*real/pic
    __s = 1/((1/F)-(1/S))
    if abs(__s -_s) >= Epsilon:
        S, __s = find_S(__s, real, pic)
    return S, __s

def find_h(p1, p2, p3, base):
    [x1, y1], [x2, y2], [x3, y3] = p1,p2,p3
    return (abs(x2 * y3 + x1 * y2 + x3 * y1 - x3 * y2 - x2 * y1 - x1 * y3))/base

def PyThm_v(a):
    if len(a) == 2:
        return (a[0]**2 + a[1]**2)**0.5
    if len(a) == 3:
        return (a[0]**2 + a[1]**2 + a[2]**2)**0.5

def PyThm(a, b):
    return (a**2+b**2)**0.5

def getVec(a, b):
    if len(a)!= len(b):
        print("Getting Vector Error!")
        return False
    if len(a) == 2:
        return [b[0]-a[0], b[1]-a[1]]
    if len(a) == 3:
        return [b[0]-a[0], b[1]-a[1], b[2]-a[2]]    

def get_close_angle_a(O, X, Y, angle, ori_diff):
    OX_V, OY_V = getVec(O, X),getVec(O, Y)
    OX, OY = length(O, X),length(O, Y)
    angle_XOY = (acos(dot(OX_V, OY_V)/(OX*OY)))
    
    diff = abs(angle-angle_XOY)
    if ori_diff < diff:
        return False, False
    print("angle_XOY = ", angle_XOY, colored(("Diff = "+str(diff)), "yellow"), end = " ")
    if diff > 10**(-5):
        # gradient = 100
        if diff > 0.1:
            gradient = 10**5
        elif diff > 0.01:
            gradient = 10**4
        elif diff > 0.001:
            gradient = 10**3
        else:
            gradient = 10**6*diff
        print(colored(str(gradient), 'red'))  
        dx, dy, dz = derivative(O, X, Y, angle)
        O[0],O[1],O[2] = O[0]+gradient*dx, O[1]+gradient*dy, O[2]+gradient*dz
        diff , O = get_close_angle_a(O, X, Y, angle, diff)
    return diff , O

def get_close_angle(O, A, B, C, angle_aOb, angle_bOc, angle_cOa):
    V_OA, V_OB, V_OC = getVec(O, A), getVec(O, B), getVec(O, C)
    L_OA, L_OB, L_OC = length(O, A), length(O, B), length(O, C)
    angle_AOB, angle_BOC, angle_COA = (acos(dot(V_OA, V_OB)/(L_OA*L_OB))), (acos(dot(V_OB, V_OC)/(L_OB*L_OC))), (acos(dot(V_OC, V_OA)/(L_OC*L_OA)))
    diff_ab, diff_bc, diff_ca = abs(angle_aOb-angle_AOB), abs(angle_bOc-angle_BOC), abs(angle_cOa-angle_COA)
    Diff = diff_ab, diff_bc, diff_ca

    # Diff = [diff_ab, diff_bc, diff_ca]
    c1, c2, c3 = 0, 0, 0
    OAB = OBC = OCA = O
    O_ans = [OAB , OBC , OCA ]
    gradient = 1
    if diff_ab > 10**(-2.5) and c1 < 200:
        c1 +=1
        dx, dy, dz = derivative(O, A, B, angle_aOb)
        OAB[0],OAB[1],OAB[2] = O[0]+gradient*dx, O[1]+gradient*dy, O[2]+gradient*dz
        diff_ab , OAB = get_close_angle_a(OAB, A, B, angle_aOb, diff_ab)
    if diff_bc > 10**(-2.5) and c2 < 200:
        c2 +=1
        dx, dy, dz = derivative(O, B, C, angle_bOc)
        OBC[0],OBC[1],OBC[2] = O[0]+gradient*dx, O[1]+gradient*dy, O[2]+gradient*dz
        diff_bc , OBC = get_close_angle_a(OBC, B, C, angle_bOc, diff_bc)
    if diff_ca > 10**(-2.5) and c3 < 200:
        c3 +=1
        dx, dy, dz = derivative(O, C, A, angle_cOa)
        OCA[0],OCA[1],OCA[2] = O[0]+gradient*dx, O[1]+gradient*dy, O[2]+gradient*dz
        diff_ca , OCA = get_close_angle_a(OCA, C, A, angle_cOa, diff_ca)
    ans_list, diff_list = [], []
    for d in range(len(Diff)):
        if Diff[d] != False:
            ans_list.append(O_ans[d])
            diff_list.append(Diff[d])
    return ans_list[diff_list.index(min(diff_list))]
    
    
def iter(diff, O, X, Y, angle):
    count, gradient = 0, 1
    


def derivative(O, X, Y, angle):
    x, y, z = symbols('x', real = True), symbols('y', real = True), symbols('z', real = True)
    DIFF = Abs(angle-(acos(((X[0]-x)*(Y[0]-x) + (X[1]-y)*(Y[1]-y)+(X[2]-z)*(Y[2]-z))/((((X[0]-x)**2+(X[1]-y)**2+(X[2]-z)**2)**0.5)*(((Y[0]-x)**2+(Y[1]-y)**2+(Y[2]-z)**2)**0.5)))))
    dx, dy, dz = diff(DIFF, x), diff(DIFF, y), diff(DIFF, z)
    dx, dy, dz = dx.subs({x:O[0], y:O[1], z:O[2]}), dy.subs({x:O[0], y:O[1], z:O[2]}), dz.subs({x:O[0], y:O[1], z:O[2]})
    return dx, dy, dz


def dot(A, B):
    return A[0]*B[0]+A[1]*B[1]+A[2]*B[2]

# Read img by openCV
start_time = time.time()
filename = sys.argv[1]
img_rgb = (cv2.imread(filename))[:,:,::-1]
# Get A, B, C's x,y-coordinate
height, width, color = img_rgb.shape
# print(height, width)
lower_A , lower_B, lower_C = np.array([254, -1, -1]), np.array([254, 99, -1]), np.array([-1, -1, 254])
upper_A , upper_B, upper_C = np.array([256, 1,1]), np.array([256, 101,1]), np.array([0, 0,256])
mask_A,mask_B,mask_C = cv2.inRange(img_rgb, lower_A, upper_A),cv2.inRange(img_rgb, lower_B, upper_B),cv2.inRange(img_rgb, lower_C, upper_C)
outputA,outputB,outputC = cv2.bitwise_and(img_rgb, img_rgb, mask = mask_A),cv2.bitwise_and(img_rgb, img_rgb, mask = mask_B),cv2.bitwise_and(img_rgb, img_rgb, mask = mask_C)
rA, rB, rC = np.where(outputA == 255),np.where(outputB == 255),np.where(outputC == 255)


# Definition (o,a,b,c: photo    A,B,C: real world)
o,a,b,c = [height/2,width/2] , [rA[0][0],rA[1][0]], [rB[0][0],rB[1][0]], [rC[0][0],rC[1][0]]
o,a,b,c = [o[1]-a[1], -(o[0]-a[0])], [a[1]-a[1], -(a[0]-a[0])], [b[1]-a[1], -(b[0]-a[0])], [c[1]-a[1], -(c[0]-a[0])]

print("\n***************\n")
print("o = ", o)
print("a = ", a)
print("b = ", b)
print("c = ", c)
print("\n***************\n")


ab, bc, ca = length(a, b)*1.6*0.001,length(b, c)*1.6*0.001,length(c, a)*1.6*0.001
A, B, C = [0,0,0], [275,58,0], [210,-83,0]
AB, BC, CA = length(A, B), length(B, C),length(C, A)
S_ab, _s_ab = find_S(F, AB, ab)
S_bc, _s_bc = find_S(F, BC, bc)
S_ca, _s_ca = find_S(F, CA, ca)
#___________________Approximation__________________#
# oab_h, obc_h, oca_h = (find_h(o, a, b, ab/(1.6*0.001)))*1.6*0.001, (find_h(o, b, c, bc/(1.6*0.001)))*1.6*0.001, (find_h(o, c, a, ca/(1.6*0.001)))*1.6*0.001
# __s_ab,__s_bc,__s_ca = (_s_ab**2-oab_h**2)**0.5, (_s_bc**2-obc_h**2)**0.5,(_s_ca**2-oca_h**2)**0.5
# print(_s_ab,_s_bc,_s_ca)
# print(__s_ab,__s_bc,__s_ca)
S, _s = (S_ab + S_ca + S_bc)/3 , (_s_ab + _s_ca + _s_bc)/3   # Approximation
# print(colored(("\n==> S = "+str(S)+" , s' = "+str(_s)+"   (unit : mm)"), 'green'))
# print("--- %s seconds ---" % (time.time() - start_time))

v_ao, v_ab, v_ac = getVec(a, o),getVec(a, b),getVec(a, c)
V_AB, V_AC = getVec(A, B),getVec(A, C)
l_ao, l_ab, l_ac = PyThm_v(v_ao),PyThm_v(v_ab),PyThm_v(v_ac)
L_AB, L_AC = PyThm_v(V_AB), PyThm_v(V_AC)

mul = (l_ab/L_AB+l_ac/L_AC)/2
_O = o/mul  #2D

# Compute angle_aOb
# photo-coordinate (default: z = 0)
a_c, b_c, c_c, o_c = [a[0]*1.6*0.001,a[1]*1.6*0.001, 0], [b[0]*1.6*0.001,b[1]*1.6*0.001, 0],[c[0]*1.6*0.001,c[1]*1.6*0.001, 0], [o[0]*1.6*0.001, o[1]*1.6*0.001, 0]
oa, ob, oc = length(o_c, a_c),length(o_c, b_c),length(o_c, c_c)
Oa, Ob, Oc = PyThm(_s, oa),PyThm(_s, ob),PyThm(_s, oc)
O_c = [o_c[0], o_c[1], _s]
v_Oa, v_Ob, v_Oc = getVec(O_c, a_c),getVec(O_c, b_c),getVec(O_c, c_c)

angle_aOb = (acos(dot(v_Oa, v_Ob)/(Oa*Ob)))  #*180/pi
angle_bOc = (acos(dot(v_Ob, v_Oc)/(Ob*Oc)))  #*180/pi
angle_cOa = (acos(dot(v_Oc, v_Oa)/(Oc*Oa)))  #*180/pi

O = [_O[0], _O[1] ,S]
print("O = ", O)
# O = [0,0,S]
ans = get_close_angle(O, A, B, C, angle_aOb, angle_bOc, angle_cOa)
print(colored(("\n==> O = "+str(ans))+"    (unit: mm)", 'green'))
# wAB = get_close_angle(O, A, B, angle_aOb)
# wBC = get_close_angle(O, B, C, angle_bOc)
# wCA = get_close_angle(O, C, A, angle_cOa)
# print(wAB)
# print(wAB,"\n",  wBC,"\n", wCA)

print("--- %s seconds ---" % (time.time() - start_time))
        
