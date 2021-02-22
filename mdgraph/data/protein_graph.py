import os
import numpy as np
import networkx as nx
import MDAnalysis as mda
from MDAnalysis.analysis import distances
from networkx.drawing.nx_agraph import graphviz_layout

from .preprocess import triu_to_full


def pdb_to_nx(
        pdb_file, 
        dcd_file=None, 
        select_string=None, 
        **kwargs): 
    """
    This function is to convert a MD simulation trajectory 
    to a graph presentation
    """
    mda_u = mda.Universe(pdb_file) 
    if select_string: 
        sel_atoms = mda_u.select_atoms(select_string)
    else: 
        sel_atoms = mda_u.atoms
    graph = mda_to_nx(sel_atoms, **kwargs) 
    return graph


def mda_to_nx(mda_atoms, cutoff=8): 
    """
    covert a mdanalysis atom group to a graph
    """
    G = nx.Graph() 
    # getting node attributes 
    node_list = []
    for i, atom in enumerate(mda_atoms): 
        node_list.append(
            (i, 
             {"ID": atom.id,
              "Name": atom.name, 
              "Mass": atom.mass,
              "resname": atom.resname, 
              "resnum": atom.resnum}
             ))
    G.add_nodes_from(node_list)

    # getting edgy properties 
    dist = triu_to_full(distances.self_distance_array(mda_atoms.positions))
    edges = np.where(dist < cutoff)
    edges = [(i, j, 1/dist[i, j]) for i, j in zip(*edges)]
    G.add_weighted_edges_from(edges)

    return G 


def plot_protein_graph(G, ax, cutoff=6, layout_type='graphviz'): 
    """
    Function to plot protein graph
    """
    elarge = [(u, v) for (u, v, d) in G.edges(data=True) 
              if d["weight"] > 1./cutoff]
    esmall = [(u, v) for (u, v, d) in G.edges(data=True) 
              if d["weight"] <= 1./cutoff]

    if layout_type == 'circular': 
        pos = nx.circular_layout(G)
    elif layout_type == 'spring': 
        pos = nx.spring_layout(G, seed=42)
    else:
        pos = graphviz_layout(G)
    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, ax=ax)

    # edges
    nx.draw_networkx_edges(G, pos, edgelist=elarge, width=6, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edgelist=esmall, width=6, alpha=0.5, 
        edge_color="y", style="dashed", ax=ax
    )

    # labels
    nx.draw_networkx_labels(
        G, pos, font_size=20, font_family="sans-serif", ax=ax)
    return ax
