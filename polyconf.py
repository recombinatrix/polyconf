import numpy as np
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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--name", default='polymer', help='system name string, used for filenames')
parser.add_argument("--nconfs", default=3)
args = parser.parse_args()

fname = '_'.join(args.name.split(' '))
nconfs = int(args.nconfs)


# load monomer template universes

# INITIATORS

CODI = mda.Universe('02_ATB_raw_output/renamed/pdb/CODI_bonds.pdb')
COLI = mda.Universe('02_ATB_raw_output/renamed/pdb/COLI_bonds.pdb')
P4DI = mda.Universe('02_ATB_raw_output/renamed/pdb/P4DI_bonds.pdb')
P4LI = mda.Universe('02_ATB_raw_output/renamed/pdb/P4LI_bonds.pdb')
P5DI = mda.Universe('02_ATB_raw_output/renamed/pdb/P5DI_bonds.pdb')
P5LI = mda.Universe('02_ATB_raw_output/renamed/pdb/P5LI_bonds.pdb')

# EXTENDERS

CODM = mda.Universe('02_ATB_raw_output/renamed/pdb/CODM_bonds.pdb')
COLM = mda.Universe('02_ATB_raw_output/renamed/pdb/COLM_bonds.pdb')
P4DM = mda.Universe('02_ATB_raw_output/renamed/pdb/P4DM_bonds_backbone.pdb')
P4LM = mda.Universe('02_ATB_raw_output/renamed/pdb/P4LM_bonds_backbone.pdb')
P5DM = mda.Universe('02_ATB_raw_output/renamed/pdb/P5DM_bonds_backbone.pdb')
P5LM = mda.Universe('02_ATB_raw_output/renamed/pdb/P5LM_bonds_backbone.pdb')

# TERMINATORS

CODT = mda.Universe('02_ATB_raw_output/renamed/pdb/CODT_bonds.pdb')
COLT = mda.Universe('02_ATB_raw_output/renamed/pdb/COLT_bonds.pdb')
P4DT = mda.Universe('02_ATB_raw_output/renamed/pdb/P4DT_bonds.pdb')
P4LT = mda.Universe('02_ATB_raw_output/renamed/pdb/P4LT_bonds.pdb')
P5DT = mda.Universe('02_ATB_raw_output/renamed/pdb/P5DT_bonds.pdb')
P5LT = mda.Universe('02_ATB_raw_output/renamed/pdb/P5LT_bonds.pdb')

monomers = {
    'CODI' : CODI ,
    'COLI' : COLI ,
    'P4DI' : P4DI ,
    'P4LI' : P4LI ,
    'P5DI' : P5DI ,
    'P5LI' : P5LI ,
    'CODM' : CODM ,
    'COLM' : COLM ,
    'P4DM' : P4DM ,
    'P4LM' : P4LM ,
    'P5DM' : P5DM ,
    'P5LM' : P5LM ,
    'CODT' : CODT ,
    'COLT' : COLT ,
    'P4DT' : P4DT ,
    'P4LT' : P4LT ,
    'P5DT' : P5DT ,
    'P5LT' : P5LT ,
}

middle = [
    'CODM',
    'COLM',
    'P4DM',
    'P4LM',
    'P5DM',
    'P5LM',
]


def first (monomer):
    # build the first monomer of a polymer
    u = monomer.copy()
    u.residues.resids = 1
    return(u)

def extend (u,monomer,n,rot=180):
    # extend a polymer u by a monomer u_, by fitting the backbone atoms (CMA_n+1, CN_n+1) to (CA_n,C_n)
    # rot is no longer used
    
    CA = u.select_atoms('name CA').positions[-1]
    C = u.select_atoms('name C').positions[-1]

    u_ = monomer.copy()
    u_.residues.resids = n+1

    CMA = u_.select_atoms(f' name CMA').positions[0]

    # first, move CMA_n+1 to CA_N

    T = CA - CMA

    u_.atoms.translate(T)

    # next, rotate around cross product of backbone vectors to align C_n to CN_n+1

    CMA = u_.select_atoms(f' name CMA').positions[0]
    CN = u_.select_atoms(f' name CN').positions[0]

    v1 = CN - CA
    v1_n = np.linalg.norm(v1)

    v2 =  C - CA
    v2_n = np.linalg.norm(v2)


    theta = degrees(np.arccos(np.dot(v1,v2)/(v1_n * v2_n)))

    k = np.cross(v1,v2)
    u_r1 = u_.atoms.rotateby(theta,axis=k,point=CA)

    # next, rotate new monomer by rot degrees  # deprecated in favour of the shuffler

    #  u_r2 = u_r1.atoms.rotateby(rot,axis=v2,point=CA)

    # combine extended polymer into new universe

    new = mda.Merge(u.atoms, u_r1.atoms)

    # add a new bond linking -C to CA
    nn = n + 1
    prev_C = new.select_atoms(f"resid {n} and name C").indices[0]
    new_CA = new.select_atoms(f"resid {nn} and name CA").indices[0]
    new.add_bonds([(prev_C,new_CA)])

    new.dimensions = list(new.atoms.positions.max(axis=0) + [0.5,0.5,0.5]) + [90]*3
    return (new)


def genconf(pol, # universe containing raw polymer conf with bond information
    n=5, # number of conformations to generate 
    cutoff=0.3, # minimum atom/atom distance for clash checker.  This is crude but should be good enough for EM
    limit=20, # how many times to generate a monomer conformation without clashes, before giving up
    verbose=False,
    fname='polymer_conf',
    ):

    errors = {}
    e = 0
    for rep in range(1,n+1):
        conf = pol.copy()  # you need to leave the dummy atoms in, or the bond index doesn't match the atom index
        
        print()
        #print(f'\nGenerating conformation {rep} of {n}')
        first = True

        for i in tqdm(conf.residues.resids,desc=f'Shuffling dihedrals, conformation {rep} of {n}'):
            prev = i-1
            
            #print(f'\nresid {i}')
            #print(conf.select_atoms(f'resid {i}').residues.resnames)

            # if not first:
            #     # shuffle -CA C bond
            #     #print('# shuffle C -CA bond')
            #     conf = shuffle(conf,sel=f"(resid {i} and name CA) or (resid {prev} and name C)")
            # # shuffle CA C bond
            # #print('# shuffle CA C bond')
            #done = False
            #while not done:
            conf,clash = shuffle(conf, sel=f"resid {i} and name CA C",    )
                #print(clash)
        #        done = clash >= cutoff
            # shuffle CA CB bond
            #print('# shuffle CA CB bond')
            #print(conf.atoms.indices)
            #print(len(conf.atoms.indices))

            # oh this is failing because of the missing dummy atoms
            # ok leave them in for now, cut when you save it
            done = False
            tries = 0
            name = conf.select_atoms(f'resid {i}').residues.resnames[0]
            while not done:
                conf,clash1 = shuffle(conf, sel=f"resid {i} and name CA CB",   mult=6 ) 
                conf,clash2 = shuffle(conf, sel=f"resid {i} and name OG1 C1A",  ) 
                conf,clash3 = shuffle(conf, sel=f"resid {i} and name C1A C1B",  ) 
                clash = min([clash1,clash2,clash3]) # this is quick and dirty, and definitely error prone
                done = clash >= cutoff
                tries += 1

                # if failed, try shuffling entire sidechain

                if tries >= 5 and name in ['P4DM','P4LM','P5DM','P5LM','P4DT','P4LT','P5DT','P5LT','P4DI','P4LI','P5DI','P5LI']:
                    if tries == 5 and verbose : print(f'\n!!! ALERT !!!\n\n Clashes remain when shuffling residue {i} {name} after 5 tries.\n Shuffling additional sidechain dihedrals\n')
                    conf,clash4 = shuffle(conf, sel=f"resid {i} and name C2A C2B", mult=6 ) # bigger space to reduce chance of clashes
                    conf,clash5 = shuffle(conf, sel=f"resid {i} and name C3A C3B", mult=6 ) # bigger space to reduce chance of clashes
                    conf,clash6 = shuffle(conf, sel=f"resid {i} and name C4A C4B", mult=6 ) # bigger space to reduce chance of clashes
                    clash = min([clash,clash4,clash5,clash6])
                    done = clash >= cutoff

                if tries >= 10 and not first:
                    conf,clash7 = shuffle(conf,sel=f"(resid {i} and name CA) or (resid {prev} and name C)", mult=6) # try shuffling the -C CA bond too
                    # generally only do this to resolve clashes; I've found that shuffling both -C to CA and CA to C in every monomer leads to tightly packed structures with many clashes

                    clash = min([clash,clash7])
                    done = clash >= cutoff

                if tries >= limit: 
                    print(f'\n\n!!!!!!!!!!!\n! WARNING !\n!!!!!!!!!!!\n\n Clashes remain when shuffling residue {i} {name} after {limit} tries.\n Atoms remain within {cutoff} angstroms.\n Continuing with this geometry.\n You may wish to check for overlapping atoms.\n')
                    errors[e]={'conf':rep,'resid':i,'name':name}
                    e+=1
                    done = True



            # if name in ['P5DM','P5LM','P5DT','P5LT','P5DI','P5LI']:
            #     conf,clash = shuffle(conf, sel=f"resid {i} and name C5A C5B", mult=6 ) # bigger space to reduce chance of clashes

                #print(clash)
            #    done = clash >= cutoff

            first=False # used for if you want to do -CA C shuffling

        save(conf,f'{fname}_{rep}.gro')

    print('\n\nFinished generating conformations\n\n')

    if e > 0:
        print('Clashes log:\n')
        for v in errors.values():
            conf,resid,name = v.values()
            print(f'When generating confromation {conf}, clash at residue {resid} {name} ')
    return


def shuffle(u,sel='name CA C',mult=3):
    # based on a tutorial by richard j gowers; http://www.richardjgowers.com/2017/08/14/rotating.html
    pair = u.select_atoms(sel)
    resmin = pair.residues.resids.min()
    resmax = pair.residues.resids.max()
    #print(pair.atoms)
    CIA = u.select_atoms('name CIA')[0].index # use the initiator alpha carbon to define the fore group
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
    aft.rotateby(rot, v, point=o.position)
    forechk = fore.select_atoms(f'(resid 1 to {resmin}) and (not name CP CQ CN CMA)') # dummy atoms can't clash
    aftchk = aft.select_atoms(f'(resid 1 to {resmax}) and (not name CP CQ CN CMA)') # dummy atoms can't clash
    dist = round(distances.distance_array(forechk.atoms.positions, aftchk.atoms.positions).min(),1)  # minimum distance between fore and aft atoms, not used for now
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
     u.select_atoms('not name CN CMA CP CQ').atoms.write(fname)  # save, excluding dummy atoms


def crudesave(u,fname='out.gro'):
    # save including dummy atoms, useful for debugging the geometry transforms
     u.atoms.write(fname)


# first, make a polymer with a known monomer composition, but with the monomers in order

polycomp = []

for m in middle:

    if m in ['CODM','COLM']:
        count = 15 # how many of each carboxyethyl monomer enantiomer
    else: count = 42 # how many of each PEG monomer enantiomer

    for i in range(count):
        polycomp += [m] # build a list with one of each monomer unit present in the final polymer

# then randomise positions of monomers

poly=random.sample(polycomp,len(polycomp)) 

poly[0] = poly[0][:-1] + 'I' # convert first monomer from middle to initiator

poly[-1] = poly[-1][:-1] + 'T' # convert last monomer from middle to terminator

# now, use the composition to build the polymer

for i in tqdm(range(len(poly)),desc='Building initial polymer geometry'):
    #print(poly[i],i)

    if i == 0:
        pol = first(monomers[poly[i]])
        #print( pol.residues.resids)
        #print(pol.select_atoms('resid 1 and name CA').positions[0])
    else:
        rot = (i*45)%360 # not used anymore; replaced with genconf shuffling
        pol = extend(pol,monomers[poly[i]],n=i,rot=rot)


save(cleanup(pol),f'{fname}_linear.gro')

genconf(pol,n=nconfs,fname=fname,verbose=False)
