# -*- coding: utf-8 -*-
"""
Visualisation of particles flowing between points on a plane. Appearing and disappearing in them. Flows ~log(x) where x are input data for all pairs.
@author: Mateusz
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.random import uniform
from gif_animate import animate
import foodwebs as fw

#we aim at smoothness and clarity of visualisation so velocities are aligned so that it looks nice
f=60 #FPS
interval=1000/f
lengthT=5 #how long should the animation be in s
frameNumber=f*lengthT #number of frames in ms
v=1/(50*lengthT) #global velocity along the lines, l=length of line

#INPUT

#A food web example
net = fw.read_from_SCOR('./examples/Alaska_Prince_William_Sound_2.dat')
flows=net.get_flow_matrix(boundary=False)
flows.loc[:,'Detritus']=np.full(len(flows),0.0)

yDf=pd.DataFrame({'y':fw.calculate_trophic_levels(net)})
tlbinsInt=list(map(lambda x: 0.5*x,list(range(1, 14,2))))
yDf['TLbinInteger'] = pd.cut(x=yDf['y'].apply(np.float), bins=tlbinsInt) #bins include the rightmost edge
yDf['TLbin']=yDf['TLbinInteger'].apply(lambda x: x.mid)
grouped=yDf.groupby('TLbin').agg('count')
maxWidth=grouped['y'].max()
minWidth=grouped['y'].min()
grouped['xs']=grouped['y'].apply(lambda x: list(np.linspace(5+5*(maxWidth-x),95-5*(maxWidth-x),x)))

def assignXandUpdate(yrow,grouped): #assign the next free x from the list of available values in yDf 
    return(grouped.loc[yrow['TLbin'],'xs'].pop(0))
yDf['x']=yDf.apply(assignXandUpdate,axis='columns', grouped=grouped)

xs=yDf['x']
ys=np.interp(yDf['y'], (yDf['y'].min(), yDf['y'].max()), (5, 95))
    
#A simpler example
# n=6 #number of points
# xs=[5,5,10,40,90]#n Series of X coordinates 
# ys=[5,50,90,40,5]#n Series of Y coordinates 
# flows=pd.DataFrame([[0,1e5,0,1e4,0],
#                     [0,0,1e3,0,0],
#                     [0,0,0,0,0],
#                     [0,0,5e3,0,0],
#                     [0,1e2,0,2e5,0]],
#                    ) #matrix of flows flows_ij=flow i -> j


#INITIALIZE FLOW
minFlow=flows[flows>0].min().min() #the minimal flow will correspond to a fixed number of particles
minPartNum=5  #fixed number of particles for a minimal flow

def calcPartNum(flow, minFlow_=minFlow, minPartNum_=minPartNum): #calculate the number of particles so that they are scaled logarithmically
    if flow==0.0: return 0.0
    return(int(minPartNum*np.log10(flow/minFlow)))

partNumber=flows.applymap(calcPartNum)

def stretch(x,l): return(np.full(int(l),x))
#initialize particles spaced randomly (uniform dist) along a line defined by start and finish
def particlesInOneFlow(nOne_, x1, x2, y1,y2):
    #points will be along a line, with s in [0,1] tracing their progress
    lx=(x2-x1)
    ly=(y2-y1)
    #we want the nOne to represent flow density, so we need to normalize to path length
    nOne=int(nOne_*np.sqrt(lx**2+ly**2)/20)
    s=uniform(0,1,nOne)
    lxd=2 #max(lx/20,3)
    distortX=uniform(-lxd,lxd,nOne)
    distortY=uniform(-lxd,lxd,nOne)#spread them randomly also in direction perpendicular to the line
    
    d={'s':s, 'x':x1+s*lx,'y':y1+s*ly, 'x1':stretch(x1,nOne)+distortX,'y1':stretch(y1,nOne)+distortY, 'lx':lx,'ly':ly}
    return (pd.DataFrame(d))

#now initialize particles
particles=pd.DataFrame()
for i in range(len(flows)):
    for j in range(len(flows)):
        if flows.iloc[i,j]!=0.0: #we do nothing for zero flows
            particles=pd.concat([particles,particlesInOneFlow(partNumber.iloc[i,j], xs[i], xs[j], ys[i],ys[j])])

#RULES TO UPDATE FRAMES
def move(particles, t=interval):
    #we move a particle along its line by updating 's'
    particles.loc[:,'s']=(particles.loc[:,'s']+v*t) % 1 #updating s and cycling within [0,1]
    #which we save translated to 'x' and 'y'
    particles.loc[:,'x']=particles.loc[:,'x1'] + particles.loc[:,'lx']*particles.loc[:,'s']
    particles.loc[:,'y']=particles.loc[:,'y1'] + particles.loc[:,'ly']*particles.loc[:,'s']
    

#Adding a shadow to each of the moving particles    
trails=5 #how many shadows should be left
shadeStep=interval/20 #what should be their time delay
def panimate(frame):
    plt.cla()
    layer(frame, 0.6,t=0.)
    T=0
    for i in range(trails): #shadow
        layer(frame, max((0.4-(i+1)*(0.35/trails)),0.05),t=-shadeStep)
        T+=shadeStep
    layer(frame, 0.8,t=T+shadeStep) #the final position
    

def layer(frame, alpha_, t=interval):
    #print('Frame with alpha '+str(alpha_)+' step '+str(t))
    move(particles, t)
    plt.scatter(particles.loc[:,'x'], particles.loc[:,'y'], s=1, alpha=alpha_, color="green",edgecolors={'none'})
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])

    
animate('NetworkFlow_noDetrital.gif', panimate, 
        frames=frameNumber, interval=interval, figsize=(6, 6))