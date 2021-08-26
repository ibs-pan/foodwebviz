# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 11:43:31 2021

Class defining the image of network - positions of nodes, sizes of nodes, density of flow and other parameters independent of the way the animation is implemented.

@author: Mateusz
"""

import io
import foodwebs as fw
import numpy as np
import pandas as pd
from foodwebs import utils
import itertools
from numpy.random import uniform
import random
from part_viz_utils import squeeze_map


def list_of_regular_x_pos(row,maxWidth): #return regularly spaced x positions for num number of items
       return(list(np.linspace(10+5*(maxWidth-row['TL']),90-5*(maxWidth-row['TL']),num=row['TL'])))
       

class netImage(object):
    '''
    Class defining the image of network:
    positions of nodes, sizes of nodes, density of flow 
    and other parameters independent of the way the animation is implemented.
    
    '''

    def __init__(self, net, withDetritus=False, k_=20, min_part_num=2, map_fun=np.log10, max_part=1000):
        '''Initialize a netImage from a foodweb net.
            Parameters
            ----------
            title : string
                Name of the foodweb.
            nodes : pandas.DataFrame with node attributes- 
                - positions of nodes in canvas [0,100] x [0,100]
                - node names
                - node biomasses
            particleNumbers : [pandas.DataFrame with numbers of particles flowing 
                from row node to column node, particles moving in imports, outflows to environment]
            
        '''
        self.title = net.title
        self.nodes = self.calc_node_attributes(net)
        self.particle_numbers = self.get_particle_numbers(net, withDetritus, min_part_num=min_part_num,map_fun=map_fun, max_part=max_part)
        #now optimize node positions based on spring layout / Fruchterman - Rheingold algorithm
        def is_positive(x):
            if x>0.0: return(1.0)
            else: return(0.0)
        
        self.nodes[['x','y']]=self._fruchterman_reingold(net.flow_matrix.applymap(is_positive), 2, HardSpheres=False, if_only_attraction=True, k=k_, hold_dim=1, pos=self.get_node_pos_as_array(), iterations=20)
        self.nodes[['x','y']]=self._fruchterman_reingold(net.flow_matrix.applymap(is_positive), 2, HardSpheres=True, if_only_attraction=False, k=k_, hold_dim=1, pos=self.get_node_pos_as_array(), iterations=50)
        #print('Number of edge intersections: '+str(self.calc_num_of_crossed_edges(net)))
     
        
    
        
    def assignXandUpdate(self, yrow, grouped_): #assign the next free x from the queue of available values in pos_df 
        a=grouped_.loc[yrow['TLbin'],'xs'].pop(0)
        return(a)
    
    def randomly_assign_X_and_Update(self, yrow, grouped_): #assign the next free x from the queue of available values in pos_df 
        return(grouped_.loc[yrow['TLbin'],'xs'].pop(random.choice(range(len(grouped_.loc[yrow['TLbin'],'xs'])))))
    
    
    def outflows_to_living(self,net): # node's system outflows to living 
        living=net.node_df[net.node_df['IsAlive']]
        flows_to_living=net.flow_matrix.loc[:,living.index]
        out=flows_to_living.sum(axis='columns') #system outflows to living nodes
        return(out)
    
    def get_grouped_prop(self, yrow,grouped_,prop): #get the width of trophic layer for this node
        return(grouped_.loc[yrow['TLbin'],prop])
    
    def assign_x_rank_in_tl_layer(self, df): #check if within TL layer the node is odd or even
       old_TLbin=df.iloc[0].loc['TLbin'] #'TLbin' is the common group identifier
       i=0
       x_rank=pd.Series(dtype=int)
       for index, row in df.iterrows():
           if row['TLbin']==old_TLbin: 
               x_rank=x_rank.append(pd.Series([i+1]))
               i+=1
           else:
               x_rank=x_rank.append(pd.Series([1]))
               i=1
               old_TLbin=row['TLbin']
       return(x_rank)
           
           
    
    
    def calc_node_layer_width_around_node(self, this_tl,df): #how many nodes have TL +- 0.5
        neigh=df[np.abs(df['TL']-this_tl)<0.25]
        return(len(neigh))
    
    def calc_tl_layer(self,n): #calculate how many trophic levels should be in one horizontal band
        #0.8 worked for n=20, 0.4 for n=100
        return(0.2 + 10/n)
    
    def aggregate_in_TL_layers(self,pos_df):
        grouped=pos_df.groupby('TLbin').agg('count')
        grouped['odd_or_even']=range(1,len(grouped)+1)
        grouped['odd_or_even']=grouped['odd_or_even'].apply(lambda x: x%2==1)
        maxWidth=grouped['width'].max()
        pos_df['x_rank']=self.assign_x_rank_in_tl_layer(pos_df).values
        pos_df['odd_or_even_layer']=pos_df.apply(self.get_grouped_prop,grouped_=grouped,prop='odd_or_even', axis='columns')
        grouped['xs']=grouped.apply(list_of_regular_x_pos,maxWidth=maxWidth,axis='columns')
        pos_df=pos_df.sort_values(by='out_to_living')
        return(grouped)
    

    
    def calc_node_attributes(self, net):
        tl_layer=self.calc_tl_layer(net.n)
        #node positions are based upon node properties, here on trophic levels
        pos_df=pd.DataFrame({'TL':fw.calculate_trophic_levels(net), 'bio':net.node_df["Biomass"], 'name':net.node_df.index }) #dataframe for positions
        #we aggregate trophic levels within integers in order to evenly distribute them horizontally
        tl_strip_divisions=list(map(lambda x: tl_layer*x,list(range(1, int(14/tl_layer)))))
        pos_df['TL_bin_end'] = pd.cut(x=pos_df['TL'].apply(np.float), bins=tl_strip_divisions) #bins include the rightmost edge
        pos_df['TLbin']=pos_df['TL_bin_end'].apply(lambda x: x.mid)
        pos_df['out_to_living']=self.outflows_to_living(net)
        pos_df['y']=np.interp(pos_df['TL'], (pos_df['TL'].min(), pos_df['TL'].max()), (5, 95))
        pos_df['width']=pos_df['TL'].apply(self.calc_node_layer_width_around_node,df=pos_df)
        pos_df=self.minimal_intersections(net, pos_df) #select horizontal node positions with minimal number of intersections between flows
        #pos_df_fin['x']=50+np.asarray(np.random.uniform(-1,1,len(pos_df))) #try all starting from the middle pos_df.apply(lambda x: self.assignXandUpdate(x, grouped_=grouped), axis='columns' )
        #but to avoid flows passing over nodes en route to others due to aligned points
        #we move the nodes at random a bit
        distortX=uniform(-8,8,len(pos_df))
        pos_df['x']=pos_df['x']+distortX
        
        
        return(pos_df)
        
    def minimal_intersections(self,net, pos_df):
        #given nodes grouped within layers of trophic levels the function
        #first generates a random but regular node placement in x variable, then 
        #computes the number of intersections for each and chooses the one with the smallest
        setups=pd.DataFrame()
        for it in range(20):
            setup=pd.Series()
            grouped=self.aggregate_in_TL_layers(pos_df)
            pos_df['x']=pos_df.apply(lambda x: self.randomly_assign_X_and_Update(x, grouped_=grouped), axis='columns' )
            setup['pos']=pos_df.copy()
            
            setup['intersections']=self.calc_num_of_crossed_edges(net,pos_df)
            setup.name=it
            setups=setups.append(setup)
            
        return(setups[setups.intersections == setups.intersections.min()]['pos'].iloc[0])
        
           
        
        
    def get_particle_numbers(self, net, withDetritus, min_part_num=2, map_fun=np.log10, max_part=1000):
         #  min_part_num: fixed number of particles for a minimal flow
        
        
        if not withDetritus: flows=net.getAllToLivingFlows()
        else: flows=net.get_flow_matrix(boundary=False)
        minFlow=flows[flows>0].min().min() #the minimal flow will correspond to a fixed number of particles
        maxFlow=flows[flows>0].max().max() #the maximal flow will correspond to the maximum number of particles we can handle
        def calcPartNum(flow, minFlow_=minFlow,maxFlow=maxFlow, min_part_num_=min_part_num): #calculate the number of particles so that they are scaled (e.g. logarithmically or with square root)
            if flow==0.0: return 0.0
            return(int(squeeze_map(flow, minFlow_, maxFlow, map_fun,min_part_num_,max_part)))
        return([flows.applymap(calcPartNum), net.node_df.Import.apply(calcPartNum),(net.node_df.Export+net.node_df.Respiration).apply(calcPartNum)])
    
    def get_xs(self): #x positions of nodes
        return(self.nodes['x'])
    
    def get_ys(self): #y positions of nodes
        return(self.nodes['y'])
    
    def set_colors(self,color_series):
        self.nodes['color']=color_series

        

    def get_node_pos_as_dict(self): #returns node positions as dict {'node 1':(x1,y1),...}
        d={}
        
        def get_row_pos_dict(row):
            return({row.name: (row['x'],row['y'])})
        for ind in self.nodes.index:
            d.update(get_row_pos_dict(self.nodes.loc[ind]))
        return(d)
    
    def get_node_pos_as_array(self): #returns node positions as array from dict
        i=0
        pos_arr=np.empty([len(self.nodes),2])
        for key, item in self.get_node_pos_as_dict().items():  
            pos_arr[i] = np.asarray(item)
            i+=1
        return(pos_arr)
    
    
    
        
#following by Marscher, adapted from NetworkX pyemma/plots/_ext/fruchterman_reingold.py  https://github.com/markovmodel/PyEMMA/blob/58825588431020d7e2a2ea57a941abc86647fc0e/pyemma/plots/_ext/fruchterman_reingold.py   
    def _fruchterman_reingold(self, A, dim=2, k=None, pos=None, fixed=None,
                              iterations=100, hold_dim=None, min_dist=0.01,HardSpheres=True, if_only_attraction=False):
        # Position nodes in adjacency matrix A using Fruchterman-Reingold
        # Entry point for NetworkX graph is fruchterman_reingold_layout()
        try:
            nnodes, _ = A.shape
        except AttributeError:
            raise RuntimeError(
                "fruchterman_reingold() takes an adjacency matrix as input")
    
        A = np.asarray(A)  # make sure we have an array instead of a matrix
    
        if pos is None:
            # random initial positions
            pos = np.asarray(np.random.random((nnodes, dim)), dtype=A.dtype)
        else:
            # make sure positions are of same type as matrix
            pos = pos.astype(A.dtype)
        
        # optimal distance between nodes
        if k is None:
            k = np.sqrt(1.0 / nnodes)
        # the initial "temperature"  is about .1 of domain area (=1x1)
        # this is the largest step allowed in the dynamics.
        t = 0.1
        # simple cooling scheme.
        # linearly step down by dt on each iteration so last iteration is size dt.
        dt = t / float(iterations + 1)
        delta = np.zeros((pos.shape[0], pos.shape[0], pos.shape[1]), dtype=A.dtype)
        # the inscrutable (but fast) version
        # this is still O(V^2)
        # could use multilevel methods to speed this up significantly
        for _ in range(iterations):
            # matrix of difference between points
            for i in range(pos.shape[1]):
                delta[:, :, i] = pos[:, i, None] - pos[:, i]
            dist_from_center_x=pos[:, 0] - np.full(nnodes,50)
            # distance between points
            distance = np.sqrt((delta**2).sum(axis=-1))
            # enforce minimum distance of min_dist
            distance = np.where(distance < min_dist, min_dist, distance)
            
            #adding a hard spheres potential - to keep nodes further apart
            if HardSpheres: sphere=distance*100*np.exp(10*(10-distance))
            else: sphere=np.zeros((distance.shape[0], distance.shape[1]),dtype=float)
            centrifugal=100*dist_from_center_x**2
                       
            # displacement "force"
            if if_only_attraction: 
                force=- A * distance / k
            else:
                force=k * k / distance**2 - A * distance / k + sphere + centrifugal
                
            
            displacement = np.transpose(np.transpose(delta) *
                                    (force))\
            .sum(axis=1)
            # update positions
            length = np.sqrt((displacement**2).sum(axis=1))
            length = np.where(length < min_dist, 0.1, length)
            delta_pos = np.transpose(np.transpose(displacement) * t / length)
            
            if fixed is not None:
                # don't change positions of fixed nodes
                delta_pos[fixed] = 0.0
            # only update y component
            if hold_dim == 0:
                
                pos[:, 1] += delta_pos[:, 1]
            # only update x component
            elif hold_dim == 1:
                
                pos[:, 0] += delta_pos[:, 0]
            else:
                
                pos += delta_pos
            # cool temperature
            t -= dt
            
            pos = self._rescale_layout(pos, [[8,92],[10, 95]])
        return pos
    
    
    def _rescale_layout(self,pos, borders):
        # rescale to (origin_,scale) in all axes
    
        # shift origin to (origin_,origin_)
        lim = np.zeros(pos.shape[1])  # max coordinate for all axes
        for i in range(pos.shape[1]):
            pos[:, i] -= pos[:, i].min()
            lim[i] = max(pos[:, i].max(), lim[i])
        # rescale to (0,scale) in all directions, preserves aspect
        for i in range(pos.shape[1]):
            pos[:, i] = np.interp(pos[:, i], [0,lim[i]], borders[i])
        return pos  
    
    
    def calc_num_of_crossed_edges(self,net,positions):
        #returns the approximate number of crossing between edges given the positions of their ends
        #approximate in order not to solve line equation -> but this does not have to prolong evaluation time greatly
        links=self.get_all_link_coords(net,positions)
        links=links.dropna()
        
        pairs=pd.DataFrame(itertools.combinations(links, 2))
        pairs['crossing']=pairs.apply(doIntersect, axis='columns')
        return(pairs['crossing'].sum())
    
        
    def get_all_link_coords(self,net,positions):
        #returns coordinates of a link as a list of ((x1,y1),(x2,y2))
        living=net.node_df[net.node_df['IsAlive']]
        flows_to_living=net.flow_matrix.loc[:,living.index]
        edges=pd.DataFrame(flows_to_living.stack()) #Series of edges
        
        def get_link_coords(row):
            #returns coordinates of a link as ((x1,y1),(x2,y2))
            if row[0]==0.0: return(None)
            (node_1,node_2)=row.name
            if node_1==node_2: return(None)
            return(tuple([tuple(positions.loc[node_1,['x','y']]), tuple(positions.loc[node_2,['x','y']])]))
        return(edges.apply(get_link_coords, axis='columns'))
    
    def get_high_low(self,points):
        #get the point that is higher
        if points[0][1]>points[1][1]: return((points[0],points[1]))
        else: return((points[1],points[0]))
        
    #a heuristic solution, placing an upper bound on whether they cross   
    def do_they_cross(self,pair):
        a=pair[0]
        b=pair[1]
        if len(a)<2 or len(b)<2: 
            print(a)
            print(b)
            return(False) #strange
        if any([a[0]==b[0],a[1]==b[0],a[0]==b[1],a[1]==b[1]]): return(False) #we do not count self-loops
        (high_a,low_a)=self.get_high_low(a)
        (high_b,low_b)=self.get_high_low(b)
        if max([x[1] for x in a])<=min([x[1] for x in b]) or max([x[0] for x in a])<=min([x[0] for x in b]) or max([x[1] for x in b])<=min([x[1] for x in a]) or max([x[0] for x in b])<=min([x[0] for x in a]):
            return(False) #they are separated in one of the variables
        if len(high_a)<2 or len(high_b)<2 or len(low_a)<2 or len(low_b)<2: 
            print(a)
            print(b)
            return(False)
        
        if np.sign((high_a[0]-high_b[0])*(low_a[0]-low_b[0])): #if the ends are in different order they most likely cross - usually the vertical length component is the same (==1, it is one trophic level)
            return(True)
        
        
def onSegment(p, q, r):
            if ( (q[0] <= max(p[0], r[0])) and (q[0] >= min(p[0], r[0])) and 
                   (q[1] <= max(p[1], r[1])) and (q[1] >= min(p[1], r[1]))):
                return True
            return False
  
def orientation(p, q, r):
    # to find the orientation of an ordered triplet (p,q,r)
    # function returns the following values:
    # 0 : Colinear points
    # 1 : Clockwise points
    # 2 : Counterclockwise
      
    # See https://www.geeksforgeeks.org/orientation-3-ordered-points/amp/ 
    # for details of below formula. 
      
    val = (float(q[1] - p[1]) * (r[0] - q[0])) - (float(q[0] - p[0]) * (r[1] - q[1]))
    if (val > 0):
          
        # Clockwise orientation
        return 1
    elif (val < 0):
          
        # Counterclockwise orientation
        return 2
    else:
          
        # Colinear orientation
        return 0
  
# The main function that returns true if 
# the line segment 'p1q1' and 'p2q2' intersect. From https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
def doIntersect(pair):
    ((p1,p2),(q1,q2))=pair
    # Find the 4 orientations required for 
    # the general and special cases
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
  
    # General case
    if ((o1 != o2) and (o3 != o4)):
        return True
  
    # Special Cases
  
    # p1 , q1 and p2 are colinear and p2 lies on segment p1q1
    if ((o1 == 0) and onSegment(p1, p2, q1)):
        return True
  
    # p1 , q1 and q2 are colinear and q2 lies on segment p1q1
    if ((o2 == 0) and onSegment(p1, q2, q1)):
        return True
  
    # p2 , q2 and p1 are colinear and p1 lies on segment p2q2
    if ((o3 == 0) and onSegment(p2, p1, q2)):
        return True
  
    # p2 , q2 and q1 are colinear and q1 lies on segment p2q2
    if ((o4 == 0) and onSegment(p2, q1, q2)):
        return True
  
    # If none of the cases
    return False
        
    
    
