import numpy as np 
from pulp import *
import pandas as pd


class order_picking:
    def __init__(self):
        pass
        
    def picking(self,items,time_matrix, vehicles,limit):

        if (len(items))==2:
            return(1, 2*time_matrix.loc[0,1])
        if (len(items))==3:
            return(1, time_matrix.loc[0,1] + time_matrix.loc[0,2] + time_matrix.loc[0,1] + time_matrix.loc[2,0])

        result = []
        result_name = []
        M=10000
        result_df = pd.DataFrame()
        row,col = time_matrix.shape
        vehicles=vehicles

        problem = LpProblem('Warehouse_Picking', LpMinimize)

        # Decision variable X & Y for picker route
        decisionVariableX = LpVariable.dicts('decisionVariable_X', ((i, j, k) for i in items for j in items for k in range(vehicles)), lowBound=0, upBound=1, cat='Integer')
        decisionVariableY = LpVariable.dicts('decisionVariable_y', ((i, k) for i in items for k in range(vehicles)), lowBound=0, upBound=1, cat='Integer')

        # subtours elimination
        decisionVariableU = LpVariable.dicts('decisionVariable_U', ((i, k) for i in items for k in range(vehicles)), lowBound=0, cat='Integer')

        # Decision variable T for picker arrival time
        decisionVariableT = LpVariable.dicts('decisionVariable_T', ((i,k) for i in items for k in range(vehicles)), lowBound=0, cat='Float')

        # Objective Function
        problem += lpSum(decisionVariableT[i, k] for i in items for k in range(vehicles))

        for k in range(vehicles):
            problem += lpSum(decisionVariableY[i, k] for i in items) <= limit
            for i in items:
                problem += (decisionVariableX[i,i, k] == 0) # elimination of (1 to 1) route
                if i==0:
                    problem += (decisionVariableT[i, k] == 0) # at node 0 time=0

        for i in items:
            if (i != 0):
                problem += lpSum(decisionVariableY[i, k] for k in range(vehicles)) == 1 # all non-zero nodes are visited once
                for k in range(vehicles):
                    problem += lpSum(decisionVariableX[i, j, k] for j in items)== decisionVariableY[i, k] 
                    problem += lpSum(decisionVariableX[j, i, k] for j in items)== decisionVariableY[i, k] 
            if (i == 0):
                for k in range(vehicles):
                    problem += lpSum(decisionVariableX[i, j, k] for j in items) <= 1 
                    problem += lpSum(decisionVariableX[j, i, k] for j in items) <= 1 
            

        for i in items:
            for j in items:
                for k in range(vehicles):
                    if i != j and (j != 0):
                        problem += decisionVariableT[j, k] >= decisionVariableT[i, k] + time_matrix.iloc[i][j] - M*(1-decisionVariableX[i,j, k]) # Calculating time of arrival at each node
                    if i != j and (i != 0) and (j != 0):
                        problem += decisionVariableU[i, k]  <=  decisionVariableU[j, k] + M * (1 - decisionVariableX[i, j, k])-1 # sub-tour elimination for picker


        status = problem.solve(CPLEX_CMD(msg=0)) 
        for var in problem.variables():
            if (problem.status == 1):
                if (var.value() !=0):
                    result.append(var.value())
                    result_name.append(var.name)
        result_df['Variable Name'] = result_name
        result_df['Variable Value'] = result

        # return
        return (problem.status, problem.objective.value(), result_df)
    
    def single_order_picking(self, on_hand_dump, time_matrix, items, pick_quantity, veh, cap):
        instance_detail = on_hand_dump.copy()
        instance_detail['time=0']=0
        instance_detail = instance_detail[['time=0','on_hand']]
        items.insert(0,0)
        pick_quantity.insert(0,0)
        on_hand=on_hand_dump.iloc[items]
        drop_list =[]
        
        for en,i in enumerate(items):
            if on_hand.loc[i,'on_hand']<pick_quantity[en]:
                drop_list.append(i)
            else:
                if (i !=0):
                    picked_quantity = pick_quantity[en]
                    on_hand.loc[i,'on_hand'] = on_hand.loc[i,'on_hand'] - picked_quantity
        for i in drop_list:
            items.remove(i)
        sts, obj,det = self.picking(items,time_matrix,veh,cap)

        if sts ==1:
            on_hand_dump.update(on_hand)
            for i in range(100):
                instance_detail.iloc[i,-1]=on_hand_dump.iloc[i,-1]
        return sts, on_hand_dump
        
    def batch_picking(self, on_hand_dump, time_matrix, item_matrix, veh, cap):
        item_list = []
        quantity_list = []
        for i in item_matrix.keys():
            for j in item_matrix[i].keys():
                item_list.append(j)
            for k in item_matrix[i].values():
                quantity_list.append(k)

        picking_quantity = pd.DataFrame({'item' : item_list,
                                        'pick_quantity': quantity_list})
        picking_quantity = picking_quantity.groupby(['item'], as_index=False).sum()
        items = picking_quantity['item']
        items = items.to_list()
        pick_quantity = picking_quantity['pick_quantity']
        pick_quantity = pick_quantity.to_list()


        instance_detail = on_hand_dump.copy()
        instance_detail['time=0']=0
        instance_detail = instance_detail[['time=0','on_hand']]
        items.insert(0,0)
        
        pick_quantity.insert(0,0)
        on_hand=on_hand_dump.iloc[items]
        drop_list =[]
        for en,i in enumerate(items):
            if on_hand.loc[i,'on_hand']<pick_quantity[en]:
                drop_list.append(i)
            else:
                if (i !=0):
                    picked_quantity = pick_quantity[en]
                    on_hand.loc[i,'on_hand'] = on_hand.loc[i,'on_hand'] - picked_quantity
        for i in drop_list:
            items.remove(i)
        sts, obj,det = self.picking(items,time_matrix,veh,cap)

        if sts ==1:
            on_hand_dump.update(on_hand)
            for i in range(100):
                instance_detail.iloc[i,-1]=on_hand_dump.iloc[i,-1]
        return sts, on_hand_dump

    def zone_batch_picking(self, on_hand_dump, time_matrix, zone_matrix, item_matrix, veh, cap):

        item_list = []
        quantity_list = []
        for i in item_matrix.keys():
            for j in item_matrix[i].keys():
                item_list.append(j)
            for k in item_matrix[i].values():
                quantity_list.append(k)

        picking_quantity = pd.DataFrame({'item' : item_list,
                                        'pick_quantity': quantity_list})
        picking_quantity = picking_quantity.groupby(['item'], as_index=False).sum()
        items = picking_quantity['item']
        items = items.to_list()
        pick_quantity = picking_quantity['pick_quantity']
        pick_quantity = pick_quantity.to_list()


        instance_detail = on_hand_dump.copy()
        instance_detail['time=0']=0
        instance_detail = instance_detail[['time=0','on_hand']]

        on_hand=on_hand_dump.iloc[items]
        drop_list =[]
        for en,i in enumerate(items):
            if on_hand.loc[i,'on_hand']<pick_quantity[en]:
                drop_list.append(i)
            else:
                if (i !=0):
                    picked_quantity = pick_quantity[en]
                    on_hand.loc[i,'on_hand'] = on_hand.loc[i,'on_hand'] - picked_quantity
        for i in drop_list:
            items.remove(i)

        zone_pd = pd.DataFrame({'item' : items})
        zone_pd = pd.merge(zone_pd, zone_matrix, left_on='item', right_on=zone_matrix.index)
        zones = zone_pd['Zone'].unique().tolist()
        for i in zones:
            picking_quantity = pd.DataFrame()
            picking_quantity = zone_pd[zone_pd['Zone'] == i]
            items=[]
            pick_quantity=[]
            items = picking_quantity['item']
            items = items.to_list()
            items.insert(0,0)
            sts, obj,det = self.picking(items,time_matrix,veh,cap)
            if sts ==1:
                on_hand_dump.update(on_hand)
                for i in range(100):
                    instance_detail.iloc[i,-1]=on_hand_dump.iloc[i,-1]
        return sts, on_hand_dump

    def wave_batch_picking(self, on_hand_dump, time_matrix, item_matrix, veh, cap):
        for i in item_matrix.keys():
            item_list = []
            quantity_list = []
            
            for j in item_matrix[i].keys():
                for k in item_matrix[i][j].keys():
                    item_list.append(k)
                for l in item_matrix[i][j].values():
                    quantity_list.append(l)

            picking_quantity = pd.DataFrame({'item' : item_list,
                                            'pick_quantity': quantity_list})
            picking_quantity = picking_quantity.groupby(['item'], as_index=False).sum()
            items = picking_quantity['item']
            items = items.to_list()
            pick_quantity = picking_quantity['pick_quantity']
            pick_quantity = pick_quantity.to_list()


            instance_detail = on_hand_dump.copy()
            instance_detail['time=0']=0
            instance_detail = instance_detail[['time=0','on_hand']]
            items.insert(0,0)
            
            pick_quantity.insert(0,0)
            on_hand=on_hand_dump.iloc[items]
            drop_list =[]
            for en,i in enumerate(items):
                if on_hand.loc[i,'on_hand']<pick_quantity[en]:
                    drop_list.append(i)
                else:
                    if (i !=0):
                        picked_quantity = pick_quantity[en]
                        on_hand.loc[i,'on_hand'] = on_hand.loc[i,'on_hand'] - picked_quantity
            for i in drop_list:
                items.remove(i)
            sts, obj,det = self.picking(items,time_matrix,veh,cap)

            if sts ==1:
                on_hand_dump.update(on_hand)
                for i in range(100):
                    instance_detail.iloc[i,-1]=on_hand_dump.iloc[i,-1]
        return on_hand_dump
