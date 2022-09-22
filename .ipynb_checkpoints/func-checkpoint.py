# Copyright (c) <2022>, <Polatucha16>

import numpy as np
from numpy import linalg as LA
import networkx as nx

from shapes import Ball, Plane

# Ball - plane geometry handling:

def center_to_plane_dist(ball: Ball, plane: Plane):
    """
        formula for distance is calculated using: |perp.ball_center - perp.plane_center|/norm(perp) 
        -> perp is vector perpendicular to plane,
        -> plane_center can be any point of the plane.
    """   
    dist = abs(
            np.vdot(plane.perp(),ball.cen)
            -np.vdot(plane.perp(),plane.cen)
                )/np.linalg.norm(plane.perp())
    return dist




def class_ball_plane_intersection(ball: Ball, plane: Plane):
    """
        Returns Ball object ball(cen,rad) representing intersection of a ball and plane in inner coordinates of plane.
            type ball.cen = np.array shape=(1,2)
            type ball.rad = float
        Parameters:
            ball -  should be  Ball(np.array shape=(1,3), float)
            plane - should be Plane(np.array shape=(1,3), np.array shape=(3,3))
        Inner coordinates of plane are set by plane.span(), example: 
            inner coordinates: (u,v)_plane represent the point: plane.cen + u*plane.span[0] + v*plane.span[1]
        Return case:
            If ball and plane have single point intersection or intersection is empty then returns 
            ball at the center (0,0) and radius 0
    """
    
        # 1. Check if the plane and the ball have nonempty intersection i.e. |n.c - p.n|/norm(n) < r ? 
    ball_result = Ball(np.array([[0,0]]), 0)
    dist = center_to_plane_dist(ball, plane)
    
    if dist >= ball.rad :
        return ball_result
    else:
        #2. Projection of the center of the ball onto Plane along plane.perp() in coordinate system given by plane.span()
        # Inner coordinates are dot products with first two vectors from the ON basis, thus
        c_rel = ball.cen - plane.cen    # center_relative is center of the ball B(c,r) relative to the center of the plane.
        ball_result.cen = np.matmul( plane.span() , np.transpose(c_rel))

        # 3. Find radius. Intervals of lenghs: center_to_plane_dist, ball.rad and ball_result.rad form a right angles triangle.
        # Therefore center_to_plane_dist^2 + ball_result.rad^2 = ball.rad^2
        ball_result.rad = np.sqrt(ball.rad**2 - dist**2)

        return ball_result
    
    
#### Building the family of balls for whitney-like decomposition

dir_array_3d = [   np.array([[1,1,1]]),
        np.array([[1,1,-1]]),np.array([[1,-1,1]]),np.array([[-1,1,1]]),
        np.array([[1,-1,-1]]),np.array([[-1,1,-1]]),np.array([[-1,-1,1]]),
        np.array([[-1,-1,-1]])
    ]

dir_list_3d = []
for el in dir_array_3d:
    dir_list_3d.append(el.flatten().tolist())

dir_tuples_3d = []
for el in dir_list_3d:
    dir_tuples_3d.append(tuple(el))

def divide_cube(G, node, directions_list):
    """ Produce dyadic tree children of a node and attach them to tree
        Returns list of produced nodes.
    """
    
    n = nx.get_node_attributes(G, "gen")[node]
    children = directions_list
    node_list = [list(node) for child in children]
    children = [[(0.5**(n+1))*_ for _ in child] for child in children]
    
    children_fin = []
    for k, child in enumerate(children):
        children_fin.append(tuple(map(sum, zip(node_list[k],children[k]))))
    children = children_fin
    
    n = 1 + n
    for child in children:
        G.add_node(child, gen= n, final= False)
        G.add_edge(node,child)
    return children

def whitney_cover(list_of_balls_to_avoid, G, cube, max_gen, balls):
    """ Builds a tree of dyadic cubes (center,radius) forming a'la whitney cover
        of cube avoiding balls from the list.
        
        Node in tree G is final if 
            it has comparable size to distance to set of ball to avoid 
                or
            its size is equal to max_gen and it is not inside list_of_balls_to_avoid
        
        Args:
        list_of_balls_to_avoid -- list of Ball objects that decomposition avoid
        G -- it a networkx tree of nodes "G.add_node((x,y,z), gen= int, final= Boole)"
        cube -- is initial place (x,y,z) with size 2^(-gen)
        max_gen -- the limit for the size of cube the size is 2^(-max_gen)
        balls -- placeholder for final balls
    """
    cube_center = np.asarray([cube])
    cube_gen = nx.get_node_attributes(G, "gen")[cube]
    cube_size = 2**(-cube_gen)
    min_dist = np.inf
    
    for i in list_of_balls_to_avoid:
        dist = np.linalg.norm(i.cen-cube_center,2)  #dist is distance between center of current ball i and center of cube 
        diam_by_2 = np.sqrt(3)*cube_size    # cube_diameter/2
        if dist+diam_by_2-i.rad < 0:                # check if cube is inside ball i ?
            G.remove_node(cube)
            return
        min_dist = min(min_dist,dist-i.rad) # distance to the boundary of the ball 
    if (min_dist > 0 and min_dist > 2*cube_size) or cube_gen == max_gen: #cube is in decomposition
        nx.set_node_attributes(G, {cube: True}, name="final")
        balls.append(Ball(np.array([cube]),cube_size))
    else:
        children = divide_cube(G,cube,dir_tuples_3d)
        for child in children:
            whitney_cover(list_of_balls_to_avoid,G,child,max_gen, balls)
        #G.remove_node(cube)
    return 