"""This file contains methods for creating and updating cluster graphs (NOT a class) using networkx library
"""

import networkx as nx
import numpy as np
import copy
from statistics import mean, stdev


def create_cg(edge_dist_array):
    """
    Given an edge distance array, returns a new cluster graph
    :param edge_dist_array: array of entries in the format [[node1, node2, dist],[...],...]
    :return: None
    """
    cg = nx.Graph()
    if len(edge_dist_array.shape) < 2 and len(edge_dist_array) == 3:
        node1 = edge_dist_array[0]
        node2 = edge_dist_array[1]
        weight = edge_dist_array[2]
        cg.add_weighted_edges_from([(node1, node2, weight)])
    else:
        append_edges_to_cg(cg, edge_dist_array)
    #print("Created cluster: ", cg.nodes())
    return cg


def append_edges_to_cg(cg, edge_dist_array):
    """
    Given an edge distance array and cluster graph, appends edges to the cluster graph
    :param cg: cluster graph
    :param edge_dist_array: array of entries in the format [[node1, node2, dist],[...],...]
    :return: updated cg
    """

    if len(edge_dist_array) <1:
        print("Warning! attempting to add empty edge_dist_array. Cluster unmodified!")
        return cg
    #print("Adding ", edge_dist_array , " to cluster ", cg.nodes())
    for entry in edge_dist_array:
        node1 = entry[0]
        node2 = entry[1]
        weight = entry[2]
        cg.add_weighted_edges_from([(node1, node2, weight)])
    return cg


def get_cg_avg_dist(cg):
    """
    Given a cluster graph, returns its average distance
    :param cg: a cluster graph
    :return: average distance of the cluster graph cg
    """
    dist = convert_cg_to_edge_dist_array(cg)
    return mean(dist[:,2])


def convert_cg_to_edge_dist_array(cg): # not sorted edge_dist_array
    """
    Given a cluster graph, extract and convert its edges to a list
    :param cg: a cluster graoh cg
    :return: edge distance array in format [(node1, node2, dist),(...),...]
    """
    edge_dist_array = []
    for edge in cg.edges():
        node1 = edge[0]
        node2 = edge[1]
        weight = cg.get_edge_data(node1, node2)['weight']
        edge_dist_array.append((node1, node2, weight))
    return np.asarray(edge_dist_array)


def check_node_adding_criteria(cg, edge_dist_array, add_node_threshold):
    """
    Returns True if a new node can be added to the cluster. A new node is represented by a
    corresponding list of edges which link the new node to the existing nodes in the cluster
    :param cg: a cluster graph
    :param edge_dist_array: a list of edges (representing a new node) to be added to the cluster
    :param add_node_threshold: A new node can be added to a cluster if the average distance of the new cluster is below
             the add_node_threshold of the cluster.
    :return: boolean
    """
    new_cg = copy.deepcopy(cg)
    new_cg = append_edges_to_cg(new_cg, edge_dist_array)
    new_cg_avg_dist = get_cg_avg_dist(new_cg)
    cg1_avg_dist = get_cg_avg_dist(cg)
    diff_dist = ((new_cg_avg_dist - cg1_avg_dist)*100)/cg1_avg_dist
    #print("Avg dist new cg: ", new_cg_avg_dist, " prev cg: ", cg1_avg_dist, " diff dist ", diff_dist)
    return diff_dist < add_node_threshold


def check_and_add_node(cg, node, edge_dist_array, add_node_threshold):
    """
    Check and add new node to the cluster if conditions by check_node_adding_criteria() are met
    :param cg: a cluster graph
    :param node: a new node
    :param edge_dist_array: a list of edges (representing a new node) to be added to the cluster
    :param add_node_threshold: A new node can be added to a cluster if the average distance of the new cluster is below
             the add_node_threshold of the cluster
    :return: None as the cluster is simply updated
    """
    avg_dist = get_cg_avg_dist(cg)
    cluster_size = len(cg.nodes())
    if cluster_size>0 and avg_dist>0:
        add_node_threshold = add_node_threshold /(cluster_size * avg_dist)
    #print("Attempting to add node", node, " to cluster ", cg.nodes())
    #print("Add Node threshold: ", add_node_threshold)
    if not cg_has_node(cg, node) and check_node_adding_criteria(cg, edge_dist_array, add_node_threshold):
        append_edges_to_cg(cg, edge_dist_array)
        #print("Node added, cluster becomes ", cg.nodes())
    else:
        pass
        #print("Node not added")


def check_and_add_node2(cg, node, edge_dist_array, add_node_threshold):
    """
    Check and add new node to the cluster if conditions by check_node_adding_criteria() are met
    :param cg: a cluster graph
    :param node: a new node
    :param edge_dist_array: a list of edges (representing a new node) to be added to the cluster
    :param add_node_threshold: A new node can be added to a cluster if the average distance of the new cluster is below
             the add_node_threshold of the cluster
    :return: None as the cluster is simply updated
    """
    avg_dist = get_cg_avg_dist(cg)
    cluster_size = len(cg.nodes())
    if cluster_size>0 and avg_dist>0:
        add_node_threshold = add_node_threshold /(cluster_size * avg_dist)
        if not cg_has_node(cg, node) and check_node_adding_criteria(cg, edge_dist_array, add_node_threshold):
            #print("Attempting to add node", node, " to cluster ", cg.nodes())
            #print("Add Node threshold: ", add_node_threshold)
            append_edges_to_cg(cg, edge_dist_array)
            #print("Node added, cluster becomes ", cg.nodes())
        else:
            pass
    else:
        if mean(edge_dist_array[:,2])< 0.3: # why
            append_edges_to_cg(cg, edge_dist_array)
            #print("I am here!", cg.nodes(), edge_dist_array)


def merge_clusters(cg1, cg2, edge_dist_array):
    """
    Merge two clusters cg1 and cg2
    :param cg1: cluster graph 1
    :param cg2: cluster graph 2
    :param edge_dist_array: a list of edges (representing a the edges connecting the the two clusters)
    :return: a new cluster cg3
    """
    cg3 = create_cg(edge_dist_array)
    cg1_edge_dist_array = convert_cg_to_edge_dist_array(cg1)
    cg2_edge_dist_array = convert_cg_to_edge_dist_array(cg2)
    append_edges_to_cg(cg3, cg1_edge_dist_array)
    append_edges_to_cg(cg3, cg2_edge_dist_array)
    append_edges_to_cg(cg3, edge_dist_array)
    return cg3


def check_cg_merge_criteria(cg1, cg2, edge_dist_array, merge_cluster_threshold):
    """
    Returns True if two clusters can be merged. The connections between the two clusters are represented by a list
    of edges
    :param cg1: cluster graph 1
    :param cg2: cluster graph 2
    :param edge_dist_array: a list of edges (representing a the edges connecting the the two clusters)
    :param merge_cluster_threshold: Two clusters can be merged together if the average distance of the new cluster is
    below the merge_thresholds for either contributing clusters
    :return: boolean
    """
    avg_dist_cg1 = get_cg_avg_dist(cg1)
    avg_dist_cg2 = get_cg_avg_dist(cg2)
    cluster_size_cg1 = len(cg1.nodes())
    cluster_size_cg2 = len(cg2.nodes())
    merge_cluster_threshold_cg1 = merge_cluster_threshold
    merge_cluster_threshold_cg2 = merge_cluster_threshold
    if cluster_size_cg1 > 0 and avg_dist_cg1>0:
        merge_cluster_threshold_cg1 = merge_cluster_threshold / (cluster_size_cg1 * avg_dist_cg1)
    if cluster_size_cg2 > 0 and avg_dist_cg2:
        merge_cluster_threshold_cg2 = merge_cluster_threshold / (cluster_size_cg2 * avg_dist_cg2)
    new_cg = merge_clusters(cg1, cg2, edge_dist_array)
    new_cg_avg_dist = get_cg_avg_dist(new_cg)
    cg1_avg_dist = get_cg_avg_dist(cg1)
    cg2_avg_dist = get_cg_avg_dist(cg2)
    if cg1_avg_dist>0 and cg2_avg_dist>0:
        diff_dist_c1 = ((new_cg_avg_dist - cg1_avg_dist)*100)/cg1_avg_dist
        diff_dist_c2 = ((new_cg_avg_dist - cg2_avg_dist)*100)/cg2_avg_dist
        #print("Avg dist c1: ",cg1_avg_dist, " c2: ", cg2_avg_dist, " c3: ",new_cg_avg_dist)
        #print("merge cluster threshold cg1: ", merge_cluster_threshold_cg1, " cg2: ",merge_cluster_threshold_cg2)
        #print(diff_dist_c1, diff_dist_c2)
        return diff_dist_c1 < merge_cluster_threshold_cg1 and diff_dist_c2 < merge_cluster_threshold_cg2
    else:
        pass
        #print("Error!! Size of cluster cannot be zero! Division by zero is attenpted")



def check_cg_merge_criteria2(cg1, cg2, edge_dist_array, merge_cluster_threshold):
    """
    Returns True if two clusters can be merged. The connections between the two clusters are represented by a list
    of edges
    :param cg1: cluster graph 1
    :param cg2: cluster graph 2
    :param edge_dist_array: a list of edges (representing a the edges connecting the the two clusters)
    :param merge_cluster_threshold: Two clusters can be merged together if the average distance of the new cluster is
    below the merge_thresholds for either contributing clusters
    :return: boolean
    """
    avg_dist_cg1 = get_cg_avg_dist(cg1)
    avg_dist_cg2 = get_cg_avg_dist(cg2)
    cluster_size_cg1 = len(cg1.nodes())
    cluster_size_cg2 = len(cg2.nodes())
    merge_cluster_threshold_cg1 = merge_cluster_threshold
    merge_cluster_threshold_cg2 = merge_cluster_threshold
    new_cg = merge_clusters(cg1, cg2, edge_dist_array)
    new_cg_avg_dist = get_cg_avg_dist(new_cg)
    if cluster_size_cg1 > 0 and cluster_size_cg2 > 0 and avg_dist_cg1>0 and avg_dist_cg2>0:
        merge_cluster_threshold_cg1 = merge_cluster_threshold / (cluster_size_cg1 * avg_dist_cg1)
        merge_cluster_threshold_cg2 = merge_cluster_threshold / (cluster_size_cg2 * avg_dist_cg2)
        diff_dist_c1 = ((new_cg_avg_dist - avg_dist_cg1)*100)/avg_dist_cg1
        diff_dist_c2 = ((new_cg_avg_dist - avg_dist_cg2)*100)/avg_dist_cg2
        #print("merge cluster threshold cg1: ", merge_cluster_threshold_cg1, " cg2: ", merge_cluster_threshold_cg2)
        #print(diff_dist_c1, diff_dist_c2)
        return diff_dist_c1 < merge_cluster_threshold_cg1 and diff_dist_c2 < merge_cluster_threshold_cg2
    elif cluster_size_cg1 > 0 and cluster_size_cg2 > 0 and avg_dist_cg1 == 0 and avg_dist_cg2 > 0:
        merge_cluster_threshold_cg2 = merge_cluster_threshold / (cluster_size_cg2 * avg_dist_cg2)
        diff_dist_c2 = ((new_cg_avg_dist - avg_dist_cg2) * 100) / avg_dist_cg2
        #print(" cg2: ", merge_cluster_threshold_cg2)
        #print(0.4, diff_dist_c2)
        #print("new avg dist", new_cg_avg_dist )
        return new_cg_avg_dist < 0.3 and diff_dist_c2 < merge_cluster_threshold_cg2
    elif cluster_size_cg1 > 0 and cluster_size_cg2 > 0 and avg_dist_cg1 > 0 and avg_dist_cg2 == 0:
        merge_cluster_threshold_cg1 = merge_cluster_threshold / (cluster_size_cg1 * avg_dist_cg1)
        diff_dist_c1 = ((new_cg_avg_dist - avg_dist_cg1) * 100) / avg_dist_cg1
        #print(" cg1: ", merge_cluster_threshold_cg1)
        #print(0.4, diff_dist_c1)
        #print("new avg dist", new_cg_avg_dist)
        return new_cg_avg_dist < 0.3 and diff_dist_c1 < merge_cluster_threshold_cg1
    else:
        pass
        #print("Error!! Size of cluster cannot be zero! Division by zero is attenpted")


def check_and_merge_cg(cg1, cg2, dist_array, merge_cluster_threshold):
    """Check and merge two clusters if conditions by check_cg_merge_criteria() are met
    :param cg1: cluster graph 1
    :param cg2: cluster graph 2
    :param edge_dist_array: a list of edges (representing a new node) to be added to the cluster
    :param merge_threshold: Two clusters can be merged together if the average distance of the new cluster is below
    the merge_thresholds for either contributing clusters
    """
    nodes = set(list(cg1.nodes.keys()) + list( cg2.nodes.keys()))
    selected_dist_array = np.asarray([e for e in dist_array if e[0] in nodes and e[1] in nodes])
    #print(selected_dist_array)
    if check_cg_merge_criteria2(cg1, cg2, selected_dist_array, merge_cluster_threshold):
        #print("Clusters ", cg1.nodes()," and ", cg2.nodes(), " are merged")
        return merge_clusters(cg1, cg2, selected_dist_array)
    else:
        pass
        #print("Clusters",cg1.nodes()," and ", cg2.nodes(), " are not merged")


def cg_has_node(cg, node):
    """
    Return True if the cluster graph contains the corresponding node
    :param cg: 
    :param node: 
    :return: 
    """
    return cg.nodes().__contains__(node)


def find_cg_having_node(cg_list, node):
    """
    From a list of cluster graphs, find the one containing a given node.
    :param cg_list: 
    :param node: 
    :return: 
    """
    for cg in cg_list:
        if cg_has_node(cg, node):
            return cg


def print_cg_list(cg_list):
    """ Print the clusters and members in a cg list
        Keyword arguments:
            cg_list: list of cluster grpahs
    """
    for cg in cg_list:
        print("\n\n")
        print("\tnodes: ",cg.nodes())
        print("\tavg. dist: ", get_cg_avg_dist(cg))


def prevent_duplicated_windows(dist_array):
    """
    Iterates over each entry in the dist_array and removes those windows which have extremely small distance,
     to prevent duplicated windows.
    :param dist_array: a flattened version of the dist_matrix in the format of entries (node0/ window0, node1/ window1, distance),
             usually sorted
    :return: a new dist_array without duplicates
    """
    duplicated_entries = [(elem[0],elem[1]) for elem in dist_array if elem[2]<0.005]
    duplicated_nodes = [item for sublist in duplicated_entries for item in sublist]
    #print(duplicated_nodes)
    selected_dist_array = []
    for entry in dist_array:
        node0, node1, weight = entry[0], entry[1], entry[2]
        if node1 not in duplicated_nodes and node0 not in duplicated_nodes:
            selected_dist_array.append(entry)
    return np.asarray(selected_dist_array)


def get_sorted_dist_array(dist_matrix, node_labels):
    """
    Given a distance matrix, create an array of entries (node0/ window0, node1/ window1, distance) sorted by distance
    :param dist_matrix: a two dimensional matrix representing the distance between windows.
    :param node_labels: labels of nodes
    :return:
    """
    edges = []
    for i in range(0, len(dist_matrix)):
        for j in range(0, len(dist_matrix)):
            if i > j:
                edges.append((node_labels[i], node_labels[j], dist_matrix[i][j]))
    sorted_dist_array = np.asarray(edges)
    sorted_dist_array = sorted_dist_array[sorted_dist_array[:, 2].argsort()]
    return sorted_dist_array


def filter_entries_containing_nodes(dist_array, nodes_set):
    """
    Filter and select entries from a dist_array containing only those nodes in the given list of nodes
    :param dist_array: a flattened version of the dist_matrix in the format of entries (node0/ window0, node1/ window1, distance),
             usually sorted
    :param nodes_set: nodes to be used for filtering
    :return: a new dist_array with entries having nodes in nodes_set
    """
    if len(dist_array)<1:
        print("Warning! dist array is empty! None is returned!")
        return None
    if len(nodes_set)<1:
        print("Warning! nodes_list is empty! None is returned!")
        return None
    selected_dist_array = []
    for entry in dist_array:
        node0, node1, weight = entry[0], entry[1], entry[2]
        if node1 in nodes_set and node0 in nodes_set:
            selected_dist_array.append(entry)
    return np.asarray(selected_dist_array)


def cluster_graph(node_labels, dist_matrix, merge_cluster_threshold = 50, add_node_threshold = 50):
    """
    Given a dist_matrix, create a list of cluster-graphs according to preset conditions
    :param node_labels: labels of nodes
    :param dist_matrix: a two dimensional matrix representing the distance between windows.
    :param merge_cluster_threshold: Two clusters can be merged together if the average distance of the new cluster is below
    the merge_thresholds for either contributing clusters
    :param add_node_threshold: A new node can be added to a cluster if the average distance of the new cluster is below
             the add_node_threshold of the cluster
    :return: a list of cluster graphs
    """
    # comment out
    #sorted_dist_array = prevent_duplicated_windows(get_sorted_dist_array(dist_matrix, node_labels))# flatten the distance matrix to a sorted distance array
    sorted_dist_array = get_sorted_dist_array(dist_matrix, node_labels)  # flatten the distance matrix to a sorted distance array
    #print("Sorted dist array")
    #print(sorted_dist_array)
    cg_list = [] # placeholder list to contain cluster_graphs
    entries_already_traversed = np.asarray([])
    ''' Placeholder list to contain the entries that have been used before they were traversed, 
    perhaps because an entry containing one of the nodes was being processed. 
    The idea is to prevent entries that were traversed from being traversed again. 
    Can be programmed differently to not require this complication
    '''
    for i in range(0, len(sorted_dist_array)): # for all entries in the sorted_dist_arr
        entry = sorted_dist_array[i]
        #print("Entries covered ", entries_already_traversed)
        if entry not in entries_already_traversed:
            #print("\n\n\n Evaluating edge ", entry)
            exclude_indices = range(0, i) # indices covered in order
            remaining_dist_array = remove_indices(sorted_dist_array, exclude_indices) #remaining dist_array after entry
            if len(cg_list) == 0: # if cg_list is empty
                cg_list.append(create_cg(entry)) # create a cluster-graph
            else:
                node0,node1,weight = entry[0],entry[1],entry[2]
                cluster_containing_node0 = find_cg_having_node(cg_list, node0)
                cluster_containing_node1 = find_cg_having_node(cg_list, node1)
                if cluster_containing_node0 is None and cluster_containing_node1 is None: # neither node0 not node 1 belong to any cluster
                    #print("Neither node ", node0 , "  nor node ", node1, " is contained by any cluster")
                    cg_list.append(create_cg(entry))
                elif cluster_containing_node0 is None and cluster_containing_node1 is not None: # node0 does not belong to cluster, node1 does. Therefore add node 0 to cluster.
                    #print("node ", node1, " is contained by cluster", cluster_containing_node1.nodes(), "while node ",node0, " is not")
                    nodes_set = set(cluster_containing_node1.nodes())
                    nodes_set.add(node0)
                    selected_dist_array = filter_entries_containing_nodes(remaining_dist_array, nodes_set)
                    entries_already_traversed = update_entry_already_traversed_list(entries_already_traversed,
                                                                                    selected_dist_array)
                    check_and_add_node2(cluster_containing_node1, node0, selected_dist_array, add_node_threshold)
                elif cluster_containing_node0 is not None and cluster_containing_node1 is None: # node0 belongs to a cluster, node1 does not. Therefore add node 1 to cluster.
                    #print("node ", node0, " is contained by cluster", cluster_containing_node0.nodes(), "while node ",node1, " is not")
                    nodes_set = set(cluster_containing_node0.nodes())
                    nodes_set.add(node1)
                    selected_dist_array = filter_entries_containing_nodes(remaining_dist_array, nodes_set)
                    entries_already_traversed = update_entry_already_traversed_list(entries_already_traversed,
                                                                                    selected_dist_array)
                    check_and_add_node2(cluster_containing_node0, node1, selected_dist_array, add_node_threshold)
                else:                                                                           # node 0 and node 1 belong to different clusters which could be merged.
                    #print("node ", node1, " is contained by cluster", cluster_containing_node1.nodes(), "while node ",node0, " is contained by cluster", cluster_containing_node0.nodes())
                    nodes_to_merge = set(list(cluster_containing_node0.nodes()) + list(cluster_containing_node1.nodes()))
                    selected_dist_array = filter_entries_containing_nodes(remaining_dist_array, nodes_to_merge)
                    entries_already_traversed = update_entry_already_traversed_list(entries_already_traversed,
                                                                                    selected_dist_array)
                    new_cluster = check_and_merge_cg(cluster_containing_node0, cluster_containing_node1,
                                                     selected_dist_array, merge_cluster_threshold)
                    if new_cluster is not None:
                        #print("merged cluster ", new_cluster.nodes(), " is created")
                        if cluster_containing_node0 in cg_list:
                            #print(cluster_containing_node0.nodes(), " is being removed!")
                            cg_list.remove(cluster_containing_node0)
                            #print("successfully removed!")
                        if cluster_containing_node1 in cg_list:
                            #print(cluster_containing_node1.nodes(), " is being removed!")
                            cg_list.remove(cluster_containing_node1)
                            #print("successfully removed!")
                        cg_list.append(new_cluster)
                    else:
                        pass
                        #print("Clusters not merged")
            #print_cg_list(cg_list)
    #print_cg_list(cg_list)
    return cg_list


def update_entry_already_traversed_list(old_traversed_array, new_entry_array):
    """
    Update dist_array by adding entries already traversed from new_entry_array while avoiding duplicates.
    Could be avoided by using sets, have to figure out numpy arr vs set usage. The entries already traversed are
    not traversed again.
    :param old_traversed_array: an array containing entries already traversed
    :param new_entry_array: entries to be appended to old_traversed_array (if not already present)
    :return:
    """
    """ 
        Keyword arguments:
            cg_list: list of cluster grpahs
    """
    #print(dist_array.shape, new_entry_array.shape)
    entries_to_add = [entry for entry in new_entry_array if entry not in old_traversed_array]
    if len(entries_to_add) > 0 :
        if len(old_traversed_array) > 0:
            return np.concatenate((old_traversed_array, entries_to_add), axis=0)
        else:
            return new_entry_array
    else:
        return old_traversed_array


def remove_indices(dist_array, indices):
    """
    Remove given indices from dist_array
    :param dist_array: a flattened version of the dist_matrix in the format of entries (node0/ window0, node1/ window1, distance),
             usually sorted
    :param indices: indices which should be removed
    :return:
    """
    selected_dist_array = [element for i, element in enumerate(dist_array) if i not in indices]
    return np.asarray(selected_dist_array)


def prune_clusters(cg_list):
    """
    ignore clusters which have high ( exceed mean + 3sd )average distance in comparison to other clusters
    :param cg_list: a list of cluster graphs
    :return: a list of clusters
    """
    #print("Pruning clusters")
    cg_avg_dist = {}
    for cg in cg_list:
        avg_dist = get_cg_avg_dist(cg)
        #print("nodes: ", cg.nodes())
        #print("avg. dist: ", avg_dist)
        cg_avg_dist[cg] = avg_dist
    sorted_avg_dist = list(cg_avg_dist.values())
    sorted_avg_dist.sort()
    #m = mean(cg_avg_dist)
    #sd = stdev(cg_avg_dist)
    remove_indexes = []
    for i in range(0,len(sorted_avg_dist)):
        if i>=3:
            m = mean(sorted_avg_dist[0:i])
            sd = stdev(sorted_avg_dist[0:i])
            for j in range(i+1, len(sorted_avg_dist)):
                j_dist = sorted_avg_dist[j]
                if j_dist > (m + 3*sd) or j_dist< (m - 3*sd):
                    #print("cg avg dist ", j_dist, ", mean: ", m, ", sd:", sd)
                    remove_indexes.append(j)
    selected_avg_dist = [item for i,item in enumerate(sorted_avg_dist) if i not in remove_indexes]
    #print(remove_indexes)
    #print(selected_avg_dist)
    selected_clusters = [cg for cg in cg_avg_dist if cg_avg_dist[cg] in selected_avg_dist]
    return selected_clusters


'''

def iterate(node_labels, dist_matrix, merge_thresholds, add_node_thresholds, prune = True): ## REDESIGN REQUIRED FOR PRUNING
    """

    :param node_labels:
    :param dist_matrix:
    :param merge_thresholds:
    :param add_node_thresholds:
    :param prune:
    :return:
    """
    """ Iterate the cluster graph algorithm over a range of merge_thresholds and add_node_thresholds to show effect of the variation
        Keyword arguments:
            node_labels: numeric labels representing the windows
            dist_matrix: a two dimensional matrix representing the distance between windows.
            merge_thresholds: an array of merge_thresholds values
            add_node_thresholds: an array of add_node_thresholds values
    """
    cg_list_list = []
    for m_threshold in merge_thresholds:
        for a_threshold in add_node_thresholds:
            cg_list = cluster_graph(node_labels, dist_matrix, merge_cluster_threshold=m_threshold,
                                    add_node_threshold=a_threshold)
            if prune:
                prune_clusters(cg_list)
            cg_list_list.append([m_threshold, a_threshold, cg_list])
    #print([(entry[0], entry[1], len(entry[2])) for entry in cg_list_list])
    return [entry[2] for entry in cg_list_list]

'''

'''
def add_node_to_cg(cg, entry, dist_array): # entry = [node1, node2, dist] dist_array = remaining dist_array not prev entries
    cg.add_weighted_edges_from(entry[0],entry[1], entry[2])
    append_edges_to_cg(dist_array, cg)

'''