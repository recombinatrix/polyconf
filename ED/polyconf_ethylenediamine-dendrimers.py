#!/usr/bin/env python 
 
import numpy as np
import pandas as pd
from tqdm import tqdm  # progress bar

import MDAnalysis as mda
from MDAnalysis.analysis import distances
from math import degrees
import networkx as nx

import random # we'll be using this to randomise monomer positions

import networkx as nx  # we'll be using this to define halves of the molecule during dihedral shuffling

# by Ada Quinn, University of Queensland

# this script takes monomer structures and extends them into a polymer by fitting to dummy atoms on each monomer
# then it generates conformations by shuffling dihedrals

# =============================================================================
# ================================ TO DO LIST =================================
# =============================================================================

# TO DO:  better clash checking, better clash handling, more generaliseable builder, more generalisable shuffler

# strategy one:  shuffle dihedrals one at a time, starting at initiator then moving to the terminator. 
# check for fails, if fail, suffle that diehdral again, if fail a set number of times restart the monomer.  if fail the entire monomer a set number of times, step backwards again

# strategy two: shuffle backbone first, checking for backbone clashes, once happy with backbone shuffle the sidechains, check for sidechain clashes
# backbone shuffle:  CA-C, +C-CA, and CA-CB. check for clashes overlaps in the backbone (C CA CB CB2 OG1 OG2 CA1)  with the current and all previous monomers.  once the backbone is fine, move on to sidechains
# sidechain shuffle:  all the dihedrals in the sidechain.   check for overlaps within the sidechain.  if it's fine, check for overlaps with the backbone, and with all previous monomers.  if it's fine, move on
# if fail, reshuffle monomer.  repeat x times
# if entire monomer fails x times, reshuffle previous monomer.  be careful about infinite loops

# =============================================================================
# ============================= END TO DO LIST ================================
# =============================================================================

def load_mon(monomer):
    mon = mda.Universe(mdict['path'][monomer])
    return(mon)

def first (monomer):
    # build the first monomer of a polymer
    u = load_mon(monomer)
    u.residues.resids = 1
    return(u)

def extend (u,monomer,n,nn,names={'P1':'CA','Q1':'C','P2':'CMA','Q2':'CN',},rot=90,joins=[('C','CA')]):
    # extend a polymer u by a monomer u_, by fitting the backbone atoms (P2, Q2) from the new monomer to (P1,Q1) from the existing residue n, 
    # then joins the monomer nn to the existing universe with a bond between each pair of atoms in joins
    # ATOM NOMENCLATURE:
    #   P1 and Q1 are atoms in monomer n
    #   P2 and Q2 are dummy atoms in monomer nn, which correspond to the atoms P1 and Q1
    # JOINS NOMENCLATURE
    #   joins contains pairs of atoms to be linked, of the form (X_n,Y_nn)
    #   X_n   is some atom in residue n, the final residue before extension
    #   Y_n+1 is some atom in residue nn, the new residue
    #   I have not tested this script with rings, and I am not sure how it will handle them
    # NB:  this function preserves all dummy atoms. removing dummy atoms during extension causes substantial problems with indexing; you need to remove them later

    P1 = u.select_atoms(f'resid {n} and name '+names['P1']).positions[-1]
    Q1 = u.select_atoms(f'resid {n} and name '+names['Q1']).positions[-1]

    u_ = load_mon(monomer)
    u_.residues.resids = nn

    P2 = u_.select_atoms(f'resid {nn} and name '+names['P2']).positions[0]

    # first, move CMA_n+1 to CA_N

    T = P1 - P2

    u_.atoms.translate(T)

    # next, rotate around cross product of backbone vectors to align C_n to CN_n+1

    P2 = u_.select_atoms(f'resid {nn} and name '+names['P2']).positions[0]
    Q2 = u_.select_atoms(f'resid {nn} and name '+names['Q2']).positions[0]

    v1 = Q2 - P2
    v1_n = np.linalg.norm(v1)

    v2 =  Q1 - P1
    v2_n = np.linalg.norm(v2)


    theta = degrees(np.arccos(np.dot(v1,v2)/(v1_n * v2_n)))

    k = np.cross(v1,v2)
    u_r1 = u_.atoms.rotateby(theta,axis=k,point=P1)

    # combine extended polymer into new universe

    new = mda.Merge(u.atoms, u_r1.atoms)

    # add new bonds linking pairs of atoms (X_n,Y_nn) 

    for pair in joins:
        X = new.select_atoms(f"resid {n} and name {pair[0]}").indices[0]
        Y = new.select_atoms(f"resid {nn} and name {pair[1]}").indices[0]
        new.add_bonds([(X,Y)])

    new.dimensions = list(new.atoms.positions.max(axis=0) + [0.5,0.5,0.5]) + [90]*3
    return (new)



def shuffle(u,n,nn,spair=['C3','C4'],mult=3,cutoff=0.9):
    # based on a tutorial by richard j gowers; http://www.richardjgowers.com/2017/08/14/rotating.html
    pair = u.select_atoms(f'(resid {n} and name {spair[0]}) or (resid {nn} and name {spair[1]})')
    #print(pair.atoms)
    CIA = u.select_atoms('name C3')[0].index # use the initiator alpha carbon to define the fore group
    bond = u.atoms.bonds.atomgroup_intersection(pair,strict=True)[0]
    #print(bond.atoms)
    g = nx.Graph()
    g.add_edges_from(u.atoms.bonds.to_indices()) # get entire polymer as a graph
    g.remove_edge(*bond.indices)     # remove the bond from the graph
    i=0
    for c in nx.connected_components(g):    
        #print(i,c)
        i+=1
    # unpack the two unconnected graphs
    a, b = (nx.subgraph(g,c) for c in nx.connected_components(g))
    # call the graph with the initiator CA atom the 'head'
    fore_nodes = a if CIA in a.nodes() else b
    fore = u.atoms[fore_nodes.nodes()]
    aft = u.atoms ^ fore
    v = bond[1].position - bond[0].position
    o = (aft & bond.atoms)[0]
    rot = random.randrange(0,mult)*int(360/mult) # rotate by a random multiplicity
    dist=0
    fail=0
    while (dist<= cutoff) and (fail < 10):
        aft.rotateby(rot, v, point=o.position)
        forechk = fore.select_atoms(f'(resid 1 to {n}) and (not name C1 C2 C5 C6 NZ)') # dummy atoms can't clash
        aftchk = aft.select_atoms(f'(resid {nn} to 9999) and (not name C1 C2 C5 C6 NZ)') # dummy atoms can't clash
        dist = round(distances.distance_array(forechk.atoms.positions, aftchk.atoms.positions).min(),1)  # minimum distance between fore and aft atoms, not used for now
        print(nn,dist)
        fail+=1
    #print(dist)
    return(u,dist)

def cleanup(u):
     # adjust box size, center polymer in box
     box = (u.atoms.positions).max(axis=0)- (u.atoms.positions).min(axis=0) + [10,10,10]
     u.dimensions = list(box)  + [90]*3
     
     cog = u.atoms.center_of_geometry(wrap=False)
     box_center = box / 2
     return (u.atoms.translate(box_center - cog))

def save(u,fname='out.gro'):
     u.select_atoms('not name C1 C2 C5 C6 NZ').atoms.write(fname)  # save, excluding dummy atoms


def crudesave(u,fname='out.gro'):
    # save including dummy atoms, useful for debugging the geometry transforms
     u.atoms.write(fname)





if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--name", default='polymer', help='system name string, used for filenames')
    # parser.add_argument("--nconfs", default=3)
    # parser.add_argument('--count', action='store_true') # specifies monomers are given by count
    # parser.add_argument('--frac', action='store_true') # specifies monomers are as fractions, requires a total length
    # parser.add_argument('--length', type=int,required=True) # total polymer legnth
    # parser.add_argument("--monomers",type=str,default='monomers.csv',help='path to csv describing monomers, expected columns are \'resname\', \'path\', \'position\', \'fill\', and at least one of \'count\' and \'frac\'') 
    # parser.add_argument('--shuffles', type=int,default=20) # specifies monomers are given by count

    # assumes monomer resnames are four letter codes, with the final letter either M for middle, T for terminator, or I for initiator)
    # count = explicit count of how many to include
    # frac = fraction of polymer that is made up of these monomers
    # path = path to coordinate file with bond information, eg pdb file with connect records
    # position = initial, middle, terminal
    # fill = after polymer is extended based on count or frac, gencomp checks if the desired  length is required
    #           if the polymer has not reached the desired length, extends by choosing randomly from monomers where fill = True







    fname = 'ED_dendrimer_branched_4'

    df = pd.read_csv('monomers.csv')
    df.set_index('resname',inplace=True)
    mdict = df.to_dict()

    cutoff=0.85
    depth=4
    mult=6


for conf in range(1,6):

    for d in range(1,depth+1):
        print(d)
        if d == 1:
            u=first('EDC0')
        elif d == 2:
            u=extend(u,'EDC1',1,21,names={'P1':'C3','Q1':'N1','P2':'C5','Q2':'NZ'},joins=[('N1','C4')])
            u,clash=shuffle(u,21,21,mult=mult,cutoff=cutoff)
            u=extend(u,'EDC2',1,22,names={'P1':'C3','Q1':'N1','P2':'C6','Q2':'NZ'},joins=[('N1','C4')])
            u,clash=shuffle(u,22,22,mult=mult,cutoff=cutoff)
            u=extend(u,'EDC5',1,25,names={'P1':'C4','Q1':'N2','P2':'C1','Q2':'NZ'},joins=[('N2','C3')])
            u,clash=shuffle(u,25,25,mult=mult,cutoff=cutoff)
            u=extend(u,'EDC6',1,26,names={'P1':'C4','Q1':'N2','P2':'C2','Q2':'NZ'},joins=[('N2','C3')])
            u,clash=shuffle(u,26,26,mult=mult,cutoff=cutoff)
#           save(cleanup(u),f'{fname}_d2_linear.pdb')
        elif d==3:

            for i in [21,22]:

                u=extend(u,'EDC1',i,int(str(i)+'1'),names={'P1':'C3','Q1':'N1','P2':'C5','Q2':'NZ'},joins=[('N1','C4')])
                
                u,clash=shuffle(u,int(str(i)+'1'),int(str(i)+'1'),mult=mult,cutoff=cutoff)

                u=extend(u,'EDC2',i,int(str(i)+'2'),names={'P1':'C3','Q1':'N1','P2':'C6','Q2':'NZ'},joins=[('N1','C4')])


                u,clash=shuffle(u,int(str(i)+'2'),int(str(i)+'2'),mult=mult,cutoff=cutoff)

            for i in [25,26]:

                u=extend(u,'EDC5',i,int(str(i)+'5'),names={'P1':'C4','Q1':'N2','P2':'C1','Q2':'NZ'},joins=[('N2','C3')])
                u,clash=shuffle(u,int(str(i)+'5'),int(str(i)+'5'),mult=mult,cutoff=cutoff)


                u=extend(u,'EDC6',i,int(str(i)+'6'),names={'P1':'C4','Q1':'N2','P2':'C2','Q2':'NZ'},joins=[('N2','C3')])
                u,clash=shuffle(u,int(str(i)+'6'),int(str(i)+'6'),mult=mult,cutoff=cutoff)

#            save(cleanup(u),f'{fname}_d3_linear.pdb')

        elif d==4:
            for i in [211,212,221,222]:

                u=extend(u,'EDT1',i,int(str(i)+'1'),names={'P1':'C3','Q1':'N1','P2':'C5','Q2':'NZ'},joins=[('N1','C4')])
                u,clash=shuffle(u,int(str(i)+'1'),int(str(i)+'1'),mult=mult,cutoff=cutoff)
                u=extend(u,'EDT2',i,int(str(i)+'2'),names={'P1':'C3','Q1':'N1','P2':'C6','Q2':'NZ'},joins=[('N1','C4')])
                u,clash=shuffle(u,int(str(i)+'2'),int(str(i)+'2'),mult=mult,cutoff=cutoff)
            for i in [255,256,265,266]:

                u=extend(u,'EDT5',i,int(str(i)+'5'),names={'P1':'C4','Q1':'N2','P2':'C1','Q2':'NZ'},joins=[('N2','C3')])
                u,clash=shuffle(u,int(str(i)+'5'),int(str(i)+'5'),mult=mult,cutoff=cutoff)

                u=extend(u,'EDT6',i,int(str(i)+'6'),names={'P1':'C4','Q1':'N2','P2':'C2','Q2':'NZ'},joins=[('N2','C3')])
                u,clash=shuffle(u,int(str(i)+'6'),int(str(i)+'6'),mult=mult,cutoff=cutoff)

#    save(cleanup(u),f'{fname}_d4_linear.pdb')

    u.residues.resnames='CD0C'
    u.residues.resids=[x+1 for x in range(0,len(u.residues.resids))]



    save(cleanup(u),f'{fname}_conf_{conf}.pdb')

#    genconf(pol,n=nconfs,fname=fname,verbose=False,limit=args.shuffles)
