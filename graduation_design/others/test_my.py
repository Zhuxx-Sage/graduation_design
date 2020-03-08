#!/usr/bin/env python
# -*- coding:utf-8 -*-

import random
import numpy as np
from graduation_design.version_1.utils import Graph, Path, Edge
from graduation_design.version_1.env_pg import TrafficEnvironment

road_network = [[0, Edge(0,0,1), Edge(1,0,2), Edge(2,0,3)],
       [Edge(3,1,0), 0, 0, Edge(4,1,3)],
       [Edge(5,2,0), 0, 0, Edge(6,2,3)],
       [Edge(7,3,0), Edge(8,3,1), Edge(9,3,2), 0]]



gra = Graph(road_network)
traffic = TrafficEnvironment(gra)




p = [4, 9, 5]
path = Path(1, 0, p)
print("路网是：" + traffic.road_network.__str__() + "\n")
print("从1出发，所有能到0的路径是（顶点序列）：" + traffic.road_network.get_Path(1, 0).__str__() + "\n")
print("从1出发，所有能到0的路径是（边序列）：" + traffic.road_network.return_all_path_roads(1, 0).__str__()+ "\n")
print("状态矩阵为（车辆数）：" + traffic.state_matrix.__str__() + "\n")
print("行为向量为（收费值）：" + traffic.action_vector.__str__() + "\n")
print("t时间，road_e上的travel time是：" + traffic.travel_time(Edge(3,1,0)).__str__() + "\n")
print("目的地是0.路径边序列为[4, 9, 5]的travel_cost是：" + traffic.travel_cost(path).__str__() + "\n")
print("从1出发到0，路径边序列为[4, 9, 5]的traffic_demand是：" + traffic.traffic_demand(1, 0, path).__str__() + "\n")
print("从1出发到0，离开road_3去往0的车辆数是：" + traffic.out_road(Edge(3,1,0), 0).__str__() + "\n")
print("从1出发到0，secondary_demand车辆数是：" + traffic.secondery_demand(1, 0).__str__() + "\n")
print("parimary_demand车辆数是：" + traffic.primary_demand().__str__() + "\n")
print("in_roads车辆数是：" + traffic.in_road(Edge(4,1,3), 0).__str__() + "\n")
print("下个状态是：" + traffic.step(traffic.action_vector).__str__() + "\n")
