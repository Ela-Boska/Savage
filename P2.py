# -*- coding: utf-8 -*-
import numpy as np 
import torch 
import matplotlib.pyplot as plt
import math
import xlsxwriter
import xlrd
from mpl_toolkits.mplot3d import axes3d

pi = math.pi
Total= 47.380508849 
d5= 13.815158896 
Alpha= 0.75 
d0 = 11.84512721225

def solution(x,a,T1,T2,alpha,t):                      # 计算 u(x,t)
    temp = torch.arange(1,1001).float()           
    mask1 = 2*(T2-T1)/pi/temp
    mask2 = (temp*pi*alpha).cos()
    mask3 = torch.sin(temp*pi*x/a).float()
    temp2 = t.view(-1,1).float()
    temp3 = -temp2 * (temp**2).view(1,-1) * pi**2/a**2
    mask4 = torch.exp(temp3)
    mask_1 = mask1 * mask2 *mask3
    mask_1 = mask_1.view(1,-1).repeat(len(t),1)
    ans = mask_1*mask4
    ans = ans.sum(-1)
    ans = ans
    ans += (T2-T1)*x/a +T1
    return ans

def Problem2(thickness,T2,time):        # 计算第2层厚度为thickness，最高温度
    total_thickness = d0 + d5 + 1.3467032917824031 + 6.0732141078674315 + 5.5e-3/2.361075976051944e-05**0.5 + thickness*1e-3/2.043973041652856e-07**0.5
    # total_thinckness 是缩放之后的总厚度
    alpha = 1-d0/total_thickness
    t = torch.arange(time+1).float() # 设置时间断点
    ans = solution(d5,total_thickness,37,T2,alpha,t)  # 计算u(x,t)
    max_temp = max(ans)                             # 计算温度最大值
    print 'Max temperature is', max_temp            
    i = 0
    if max_temp>44:
        while True:
            if ans[i] >44.0:                        # 寻找达到44度的位置
                break
            i += 1
        print 'temperature reaches 44 at {} s'.format(i)
        print 'temperature is higher than 44 for {} s'.format(time-i)
    plt.plot(i*np.ones(100),np.linspace(37,44.3,100))
    plt.plot(t.numpy(),ans.numpy())                 # 画图
    plt.xlabel('Time/(s)')
    plt.ylabel('Temperature/(Degree)')
    plt.show()
        




def plot(alpha):
    ratio = 0.3
    sheet = xlrd.open_workbook('CUMCM-2018-Problem-A-Chinese-Appendix.xlsx').sheets()[1]
    X = sheet.col_values(0)[2:int(ratio*(len(sheet.col_values(0))-2))+2]
    Y = sheet.col_values(1)[2:int(ratio*(len(sheet.col_values(0))-2))+2]
    X = torch.tensor(X,dtype =torch.float)
    Y = torch.tensor(Y,dtype =torch.float)
    a = 21.720222740795744/(alpha-(48.08-37)/(75-37))
    x = (48.08-37)/(75-37) * a
    print 'a=',a,'x=',x, 'alpha=',alpha
    Predict = solution(x,a,37,75,alpha,X)
    def Mean_error(tensor1,tensor2):
        return torch.mean((tensor1-tensor2).abs())
    print 'Mean Error =',Mean_error(Predict,Y)

    plt.plot(X.numpy(),Predict.numpy(),'+',label='Predicted')
    plt.plot(X.numpy(),Y.numpy(),"x",label='Ground Turth')
    plt.xlabel('Time/(s)')
    plt.ylabel('Temperature/(Degree)')
    plt.legend()
    plt.show()

def Distribution():
    """
    From experiences before we have learned that a= 47.380508849 x= 13.815158896 alpha= 0.75
    The range of x involved in coumputing is [13.815158896,35.53538163675]
    We will divide it into 500 points
    """
    import time
    t1 = time.time()
    alpha = 0.75
    sheet = xlrd.open_workbook('CUMCM-2018-Problem-A-Chinese-Appendix.xlsx').sheets()[1]
    ts = sheet.col_values(0)[2:]
    thickness = 0.6+6+3.6+5
    ts = torch.tensor(ts)
    xs = torch.linspace(0,thickness,500)
    def transfer(x):
        x = x
        k1,k2,k3,k4 = 1.9849915274751876e-07,2.043973041652856e-07,3.513725392209836e-07,2.361075976051944e-05
        c1,c2,c3,c4=1000*k1**0.5,1000*k2**0.5,1000*k3**0.5,1000*k4**0.5
        thick0 = 47.380508849*0.75-21.720222740795744
        thick1 = 5/c4 +thick0
        thick2 = thick1+3.6/c3
        thick3 = thick2+6/c2
        mask1 = ((0<=x) * (x<=5)).float()
        mask2 = ((5<x) * (x<=8.6)).float()
        mask3 = ((8.6<x) * (x<=14.6)).float()
        mask4 = ((14.6<x) * (x<=15.2)).float()
        ans1 = mask1*(x/c4+thick0)
        ans2 = mask2*((x-5)/c3+thick1)
        ans3 = mask3*((x-8.6)/c2+thick2)
        ans4 = mask4*((x-14.6)/c1+thick3)
        return ans1+ans2+ans3+ans4
    ans = torch.zeros((len(ts)+1,501))
    ans[0,1:] = xs
    ans[1:,0] = ts
    Xs = transfer(xs)
    for i in range(500):
        x = Xs[i].item()
        ans[1:,i+1] = solution(x,47.380508849,37,75,0.75,ts)
    
    xlsx = xlsxwriter.Workbook('Problem1.xlsx')
    worksheet = xlsx.add_worksheet()
    for i in range(ans.size(0)):
        for j in range(ans.size(1)):
            worksheet.write(i,j,float(ans[i,j]))
    worksheet.write(0,0,'Time(s)')
    xlsx.close()
    t2 = time.time()
    print 'Finished in {} seconds'.format(t2-t1)

def Plot3d(rstride=20,cstride=1,ratio=0.4,stept = 30,stepx = 10):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sheet = xlrd.open_workbook('Problem1.xlsx').sheets()[0]
    ts = np.array(sheet.col_values(0)[1:int(ratio*(sheet.nrows-1))+1])
    xs = np.array(sheet.row_values(0)[1:])
    lent = len(ts)
    lenx = len(xs)
    ts = ts[np.arange(0,lent,stept,dtype=np.int)]
    xs = xs[np.arange(0,lenx,stepx,dtype=np.int)]
    ts = np.array(ts).reshape(-1,1).repeat(len(xs),1)
    xs = np.array(xs).reshape(1,-1).repeat(len(ts),0)
    zs = np.zeros_like(ts)
    for i in np.arange(0,lenx,stepx):
        zs[:,int(i/stepx)] = np.array(sheet.col_values(i+1))[np.arange(1,lent+1,stept).astype(np.int)]
    
    ax.plot_wireframe(ts,xs,zs,rstride=rstride,cstride=cstride)
    ax.set_xlabel('time(s)')
    ax.set_ylabel('displace(mm)')
    ax.set_zlabel('Temperature(degree)')
    plt.show()


if __name__=='__main__':
    import sys
    if len(sys.argv) ==2:
        alpha =float(sys.argv[1])
    elif len(sys.argv) ==1:
        alpha = 0.75
    else:
        print 'Usage: python P1.py alpha or python P1.py (default alpha 0.75)'
        exit()
    plot(alpha)