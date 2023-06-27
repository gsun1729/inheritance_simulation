import time
from copy import deepcopy
import time
import itertools
import numpy as np
import random

import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from scipy.spatial.distance import cdist
from itertools import combinations
from typing import List, Tuple
import pickle
import matplotlib
from matplotlib import animation
from matplotlib.patches import Circle, Rectangle
import matplotlib.pyplot as plt


from lib.structures import Particle, Box
from lib.kinetics import VolumeBasedMovement
from lib.probability import ProbOperationVol
from lib.graph import containsClique, simplifyDG
from lib.file_utils import printProgressBar
from configs.structural_config import DIFFERENCE_EPSILON, FUSION_EPSILON, FISSION_EPSILON
from configs.graphics_config import (
    BLUE_COLOR, GREY_COLOR, LINEWIDTH, TEXT_COLOR, PARTICLE_LABEL_SIZE, SPINE_ALPHA)


class Simulation:
    ParticleClass = Particle

    def __init__(self,
                 particle_radius: float,
                 n_particles: int,
                 percent_agg: float,
                 min_interaction_dist: float,
                 environment: Box,
                 diffusion_nonAgg_func: VolumeBasedMovement,
                 diffusion_Agg_func: VolumeBasedMovement,
                 p_event_nonAgg: ProbOperationVol,
                 p_event_Agg: ProbOperationVol,
                 targeted_fis_probability: float,
                 indivisible_aggs: bool,
                 dt: float):

        # Box object containing parameters for the simulation space.
        self.sim_boundaries = environment
        # Number of particles to seed simulation wtih
        self.n_particles = n_particles
        # Percentage of particles to be designated as aggregate containing
        self.percent_agg = percent_agg
        # Radius of each particle
        self.default_particle_radius = particle_radius
        self.default_particle_diameter = particle_radius * 2
        self.min_interaction_dist = min_interaction_dist

        # Functions for diffusion
        self.diffusion_nonAgg_func = diffusion_nonAgg_func
        self.diffusion_Agg_func = diffusion_Agg_func
        # Functions for fission/fusion probabilities.
        self.p_event_nonAgg = p_event_nonAgg
        self.p_event_Agg = p_event_Agg

        self.targeted_fis_probability = targeted_fis_probability
        self.indivisible_aggs = indivisible_aggs

        self.time_interval = dt
        self.timestamp = 0
        self.nSteps = 0
        self.nAgg_history = []
        self.network_history = {}
        self._init_particles()
        
        self.transition_time = None
        self.transition_snapshot = None
        
        self.DEBUG_FIRST_FUSION = False

    def _generateParticle(self, radius: float, has_agg: bool) -> None:
        """Creates a particle of radius r and aggregate status has_agg
        at a random X and Y position, with a random velocity.

        Args:
            radius (float): radius of particle
            has_agg (bool): set to True if the particle harbors aggregates
        """
        particle_ID = len(self.particle_dict)
        rand_X = np.random.uniform(low=self.sim_boundaries.lower_x + self.default_particle_radius,
                                   high=self.sim_boundaries.upper_x - self.default_particle_radius)
        rand_Y = np.random.uniform(low=self.sim_boundaries.lower_y + self.default_particle_radius,
                                   high=self.sim_boundaries.upper_y - self.default_particle_radius)
        # Assign random velocity to each particle set within range of -0.5 to 0.5 um/s
        rand_vx = np.random.uniform(-0.5, 0.5)
        rand_vy = np.random.uniform(-0.5, 0.5)
        temp_particle = self.ParticleClass(ID=particle_ID,
                                           # np.array([3.5, 3.5]),
                                           pos_vec=np.array([rand_X, rand_Y]),
                                           # np.array([-0.5,-0.5]),
                                           vel_vec=np.array(
                                               [rand_vx, rand_vy]),
                                           has_agg=has_agg,
                                           radius=radius)
        # if there are no particles present in the simulation, skip checks and add to list.
        if not self.particle_dict:
            self.particle_dict[particle_ID] = temp_particle
            return True
        # Otherwise, check all other particles in the simulation and if temp_particle collides with other
        # particles, reject.
        for p2 in self.particle_dict.values():
            if p2.checkParticleCollision(temp_particle):
                break
        else:
            # Add particle to simulation
            self.particle_dict[particle_ID] = temp_particle
            return True
        return False

    def _init_particles(self):
        """Generate the number of particles necessary for the simulation.
        """
        agg_statuses = [True] * int(self.n_particles * self.percent_agg)
        agg_statuses += [False] * (self.n_particles - len(agg_statuses))
        assert len(
            agg_statuses) == self.n_particles, "status array not n_particles long"
        random.shuffle(agg_statuses)

        self.particle_dict = {}
        # Generate particles
        for agg_stat in agg_statuses:
            while not self._generateParticle(radius=self.default_particle_radius,
                                             has_agg=agg_stat):
                pass
        # Generate network structure.
        self.network = nx.Graph()
        for particle_ID, particle_element in self.particle_dict.items():
            assert particle_ID == particle_element.ID, f"Mismatch between pIDs {particle_ID} and {particle_element.ID}"
            self.network.add_node(particle_element.ID,
                                  pos=particle_element.position,
                                  has_agg=particle_element.has_agg,
                                  obj=particle_element)

        node_list = list(self.network.nodes)
        node_combos = combinations(node_list, r=2)
        for src_nodeID, dst_nodeID in node_combos:
            src_node = self.network.nodes.get(src_nodeID)
            dst_node = self.network.nodes.get(dst_nodeID)

            src_pos = src_node.get('pos')
            dst_pos = dst_node.get('pos')

            separation = np.linalg.norm(dst_pos - src_pos)
            radii_sum = src_node.get('obj').radius + dst_node.get('obj').radius
            if separation <= (radii_sum + DIFFERENCE_EPSILON):
                self.network.add_edge(src_nodeID, dst_nodeID, sep=separation)
                # Remove edge if it forms a clique
                if containsClique(self.network):
                    self.network.remove_edge(src_nodeID, dst_nodeID)

        # populate history network
        self.history = nx.DiGraph()
        self.historyID = 0
        for subnet_member_IDs in [c for c in nx.connected_components(self.network)]:
            self.history.add_node(self.historyID,
                                  members=subnet_member_IDs,
                                  has_agg=self.getSubnetAggStatus(
                                      subnet_member_IDs),
                                  timestamp=self.timestamp)
            self.historyID += 1

    def count_nAggClusters(self):
        """Determine the number of aggregate clusters within the cell simulation

        Returns:
            (int): number of aggregate clusters
        """
        temp = deepcopy(self.network)
        for nID in self.particle_dict.keys():
            if not self.particle_dict[nID].status:
                temp.remove_node(nID)
        return nx.number_connected_components(temp)

    def assign_cluster_agg(self, seed_pID: int, size: int):
        """Randomly assigns a cluster of aggregate particles within the network structure
        Does not generate new particles, only re-assigns statuses to particles
        Args:
            seed_pID (int): particle ID of the seed to form the aggregate from
            size (int): size fo the cluster aggregate
        """
        possible_nodes = list(nx.bfs_edges(
            self.network, seed_pID, depth_limit=size*2))
        flattened_possible_nodes = [
            item for sublist in possible_nodes for item in sublist]
        # Handle cases where it is a single particle
        if not flattened_possible_nodes:
            return
        # Generate aggregate cluster, with initial seed
        selected_nodes = [flattened_possible_nodes.pop(0)]
        # Have to do while loop since bfs sometimes will register the same node twice
        # np.unique will sort array.
        try:
            while len(selected_nodes) < size:
                id = flattened_possible_nodes.pop(0)
                if id not in selected_nodes:
                    selected_nodes += [id]
        except IndexError:
            return
        # Update network by setting aggregate cluster
        for value in selected_nodes:
            self.particle_dict[value].status = True
            self.network.nodes[value]["has_agg"] = True

    def set_allNonagg(self):
        """Sets all nodes in simulation to non-aggregate containing.
        """
        for k in self.particle_dict.keys():
            self.particle_dict[k].status = False
            self.network.nodes[k]["has_agg"] = False

    def assign_n_cluster_agg(self, n_clusters: int):
        """Generate n_cluster aggregates.

        Args:
            n_clusters (int): Number of aggregate clusters to assign.
        """
        n_agg_nodes = int(self.n_particles*self.percent_agg)
        size, remainder = divmod(n_agg_nodes, n_clusters)
        sizes = [size] * n_clusters
        sizes[0] += remainder

        while True:
            # sample n_clusters seeds from all nodes in the network.
            seeds = random.sample(list(self.particle_dict.keys()), n_clusters)
            neighbors = [n for a in (
                map(lambda x: list(self.network.neighbors(x)), seeds)) for n in a]
            # Check to see if any seeds are within reach of each other; if yes, repeat seed finding

            if any(item in seeds for item in neighbors):
                continue

            for seed, agg_size in zip(seeds, sizes):
                self.assign_cluster_agg(seed_pID=seed, size=agg_size)
            if self.count_nAggClusters() != n_clusters:
                # Reset all nodes to non-agg
                for k in self.particle_dict.keys():
                    self.particle_dict[k].status = False
                continue
            else:
                break

    def getConnectedNodes(self, query_pID: int) -> List[int]:
        """Given a particle's ID, determine the IDs of other particles
        connected to it

        Args:
            query_pID (int): particle ID

        Returns:
            List[int]: list of particle IDs of particles connected to query_pID
        """
        return list(nx.node_connected_component(self.network, query_pID))

    def getSubnetAggStatus(self, ID_list: List[int]) -> bool:
        """Given a particle ID list, determine whether the structure represented
        by the particle ID list contains an aggregate within at least one of the particles  

        Args:
            ID_list (List[int]): list of particle IDs that represent the structure

        Returns:
            bool: True if the structure contains an aggregate.
        """
        for nodeID in ID_list:
            if self.network.nodes.get(nodeID).get('obj').has_agg:
                return True
        return False

    def getNetworkNeighbors(self, list_pIDs: List[int]) -> List[int]:
        """Given a list of particle IDs part of a structure, determine what other particles
        present in the simulation are within contact range equivalent to the sum of the 
        query particle's radius, the other particle's radius, and the min_interaction_dist, which
        determines the minimum range where fusion can occur.

        Args:
            list_pIDs (List[int]): list of particle IDs (ints) that are part of the network.
                The elements in list_pIDs must be part of the same continguous structure.

        Returns:
            List[int]: List of particle IDs of particles that are within range of the network specified by list_pIDs
        """

        # check if elements in list_pIDS are connected
        structure_pID_combos = list(combinations(list_pIDs, r=2))
        paths = [nx.has_path(self.network, src, dst)
                 for src, dst in structure_pID_combos]
        assert all(paths), "Elements in list_pIDs arg must be connected."

        valid_neighbors = []
        for query_particleID in list_pIDs:
            for other_particleID in self.particle_dict.keys():
                # If the other particleID is the same as the query particle ID (in list_pIDs), continue
                if other_particleID in list_pIDs:
                    continue

                # Get particle and its position, and associated reflected and wrapped positions
                query_pos_assignments = self.particle_dict.get(
                    query_particleID).getAllPositions()
                other_pos_assignments = self.particle_dict.get(
                    other_particleID).getAllPositions()

                # Returned tuple contains position type in first column, actual position in second column
                # separate into separate variables
                query_pos_types_only = list(zip(*query_pos_assignments))[0]
                query_pos_only = list(zip(*query_pos_assignments))[-1]
                other_pos_types_only = list(zip(*other_pos_assignments))[0]
                other_pos_only = list(zip(*other_pos_assignments))[-1]

                # compute the distances between sets of points for query and other particle
                distance_mx = cdist(query_pos_only, other_pos_only)
                # Determine minimum separation
                min_dist = np.min(distance_mx)
                query_minpos_type_idx, other_minpos_type_idx = np.argwhere(
                    distance_mx == min_dist)[0]

                query_minpos_type = query_pos_types_only[query_minpos_type_idx]
                other_minpos_type = other_pos_types_only[other_minpos_type_idx]

                # Check if minimum separation is within interaction distance, if valid,
                # add to valid neighbors
                if min_dist <= (self.particle_dict.get(query_particleID).radius
                                + self.particle_dict.get(other_particleID).radius
                                + self.min_interaction_dist):
                    valid_neighbors.append([min_dist,
                                            (query_particleID, query_minpos_type),
                                            (other_particleID, other_minpos_type)])
        # sort neighbors from smallest distance to largest
        valid_neighbors.sort(key=lambda row: (row[0]), reverse=False)
        return valid_neighbors

    def getNetNetContactSite(self, obj1_list: List[int], obj2_list: List[int]) -> Tuple[Tuple[int, str]]:
        """Given a list of particle IDs that represent structure 1 and a list of particle IDs that represent
        structure 2, determine whether the two structures are in collision with one another.

        Args:
            obj1_list (List[int]): list of particle IDs for structure 1
            obj2_list (List[int]): list of particle IDs for structure 2

        Returns:
            Tuple[Tuple[int, str]]: particle ID of structure 1 that is in contact with 
                associated particle ID of structure 2 provided in tuple form as 
                ((pID1, pID1_positionType), (pID2, pID2_positionType)).
                where *_positionType is of one of the following (p, ul, ll, ur, lr, w), which 
                indicates which associated position was used for the particle position

                Returns (None, None) if no contact exists
        """
        possible_interactions = itertools.product(obj1_list, obj2_list)
        for obj1_sub_pID, obj2_sub_pID in possible_interactions:
            node1 = self.particle_dict.get(obj1_sub_pID)
            node2 = self.particle_dict.get(obj2_sub_pID)

            node1_pos_assignments = node1.getAllPositions()
            node2_pos_assignments = node2.getAllPositions()
            # Get node associated position types
            node1_pos_types_only = list(zip(*node1_pos_assignments))[0]
            node2_pos_types_only = list(zip(*node2_pos_assignments))[0]
            # get node associated positions
            node1_pos_only = list(zip(*node1_pos_assignments))[-1]
            node2_pos_only = list(zip(*node2_pos_assignments))[-1]

            # compute the distances between sets of points for query and other particle
            separations = cdist(node1_pos_only, node2_pos_only)
            # Determine minimum separation
            min_dist = np.min(separations)
            (node1_minpos_type_idx, node2_minpos_type_idx) = np.argwhere(
                separations == min_dist)[0]

            node1_minpos_type = node1_pos_types_only[node1_minpos_type_idx]
            node2_minpos_type = node2_pos_types_only[node2_minpos_type_idx]

            if min_dist <= (node1.radius + node2.radius + DIFFERENCE_EPSILON):
                return ((obj1_sub_pID, node1_minpos_type),
                        (obj2_sub_pID, node2_minpos_type))
        return None, None

    def whichIsBigger(self, obj1_list: List[int], obj2_list: List[int]) -> List[int]:
        """Given two structures represented as a list of particle IDs (ints), 
        determine which structure is the larger of the two in mass, which is an attribute of 
        each particle. Mass is calculated as the sum of all of the particle masses in each 
        structure.
        If both structures are the same size, randomly return the ids of one or the other.

        Args:
            obj1_list (List[int]): list of particle IDs for object 1
            obj2_list (List[int]): list of particle IDs for object 2

        Returns:
            List[int]: list of particle IDs for the larger object
        """
        mass1 = np.sum([self.particle_dict[pID].mass for pID in obj1_list])
        mass2 = np.sum([self.particle_dict[pID].mass for pID in obj2_list])
        if mass1 > mass2:
            return obj1_list
        elif mass2 > mass1:
            return obj2_list
        else:
            return random.choice([obj1_list, obj2_list])

    def updatePositionsAdditive(self, pID_list: List[int], delta_Pxy: np.ndarray) -> None:
        """Updates all the positions of particles with IDs in in pID_list with delta_Pxy
        additively.

        Args:
            pID_list (List[int]): list of particle IDs
            delta_Pxy (np.ndarray): 2D numpy array containing position update.
        """
        for pID in pID_list:
            # update position for each
            self.particle_dict[pID].position += delta_Pxy
            self.particle_dict[pID].checkIsWrappedReflectedAndUpdate(self.sim_boundaries.upper_x,
                                                                     self.sim_boundaries.lower_x,
                                                                     self.sim_boundaries.upper_y,
                                                                     self.sim_boundaries.lower_y)

            # Deprecated, use below line instead of checkIsWRappedAndUpdate for reflective boundary
            # self.particle_dict[pID].position = np.mod(self.particle_dict[pID].position,
            #                                           [self.sim_boundaries.upper_x, self.sim_boundaries.upper_y])

            # Do not update self.network nodes with information; this is already done
            # by updating self.particle_dict as self.network entry points to self.particle_dict
            # entry.

    def updateVelocitiesMultiply(self, pID_list: List[int], alpha_Vxy: np.ndarray) -> None:
        """Updates all of the velocities of particles with IDs in pID list with alpha_Vxy
        multiplicatively

        Args:
            pID_list (List[int]): list of particle IDs
            alpha_Vxy (np.ndarray): 2D numpy array containing velocity update.
        """
        for pID in pID_list:
            self.particle_dict[pID].velocity *= alpha_Vxy
            # Do not update self.network nodes with information; this is already done
            # by updating self.particle_dict as self.network entry points to self.particle_dict
            # entry.

    def replaceVelocities(self, pID_list: List[int], new_Vxy: np.ndarray) -> None:
        """Updates all of the velocities of particles with IDs in pID list with new_Vxy
        by replacing them

        Args:
            pID_list (List[int]): list of particle IDs
            new_Vxy (np.ndarray): 2D numpy array containing velocity update.
        """
        for pID in pID_list:
            self.particle_dict[pID].velocity = new_Vxy
            # self.network.nodes[pID]['obj'].velocity = new_Vxy

    def change_velocities(self, obj1_list: List[int], obj2_list: List[int]) -> None:
        """
        structures obj1 and obj2 have collided; update their velocities and positions
        as if they had collided elastically; all particles in obj1 and obj2 are updated 
        respectively post collision to preserve a rigid structure.
        Args:
            obj1_list (List[int]): list of particle IDs associated with structure 1
            obj2_list (List[int]): list of particle IDs associated with structure 2

        """
        obj1_mass = np.sum(
            [self.particle_dict[sub_pID].mass for sub_pID in obj1_list])
        obj2_mass = np.sum(
            [self.particle_dict[sub_pID].mass for sub_pID in obj2_list])

        total_mass = obj1_mass + obj2_mass

        (obj1_contact_pID, obj1_posType), (obj2_contact_pID, obj2_posType) = self.getNetNetContactSite(
            obj1_list, obj2_list)

        obj1_contact_particle = self.particle_dict.get(obj1_contact_pID)
        obj2_contact_particle = self.particle_dict.get(obj2_contact_pID)

        obj2_selected_pos = obj2_contact_particle.getPositionByName(
            obj2_posType)
        obj1_selected_pos = obj1_contact_particle.getPositionByName(
            obj1_posType)
        sep_vector = obj2_selected_pos - obj1_selected_pos

        sep_magnitude = np.linalg.norm(sep_vector)

        # Make unit vector
        sep_vector /= sep_magnitude
        # scale to separation distance of radii sum and DIFFERENCE_EPSILON
        sep_vector *= (obj1_contact_particle.radius +
                       obj2_contact_particle.radius + DIFFERENCE_EPSILON)

        # Update structure 2 position based on structure1 position
        # print(obj1_list, obj2_list)
        obj1_list = list(obj1_list)
        obj2_list = list(obj2_list)

        update_vec = obj1_selected_pos + sep_vector - obj2_selected_pos
        # print("pre-update")
        # print(self.particle_dict[obj1_list[0]].position, self.particle_dict[obj2_list[0]].position)
        # print(self.network.nodes[obj1_list[0]]['pos'],  self.network.nodes[obj1_list[0]]['obj'].position)
        # print(self.network.nodes[obj2_list[0]]['pos'],  self.network.nodes[obj2_list[0]]['obj'].position)
        # print("update")
        # print(update_vec)
        self.updatePositionsAdditive(pID_list=obj2_list, delta_Pxy=update_vec)
        # print("post update")
        # print(self.particle_dict[obj1_list[0]].position, self.particle_dict[obj2_list[0]].position)
        # print(self.network.nodes[obj1_list[0]]['pos'],  self.network.nodes[obj1_list[0]]['obj'].position)
        # print(self.network.nodes[obj2_list[0]]['pos'],  self.network.nodes[obj2_list[0]]['obj'].position)
        d = np.linalg.norm(obj1_selected_pos - obj2_selected_pos)**2

        pos12_diff = obj1_selected_pos - obj2_selected_pos
        vel12_diff = obj1_contact_particle.velocity - obj2_contact_particle.velocity

        u1 = obj1_contact_particle.velocity - 2*obj2_mass / \
            total_mass * np.dot(vel12_diff, pos12_diff) / d * pos12_diff
        u2 = obj2_contact_particle.velocity - 2*obj1_mass / total_mass * \
            np.dot(-vel12_diff, -pos12_diff) / d * (-pos12_diff)

        self.replaceVelocities(pID_list=obj1_list, new_Vxy=u1)
        self.replaceVelocities(pID_list=obj2_list, new_Vxy=u2)

    def handle_collisions(self):
        """Detect and handle any collisions between the Particles.

        When two Particles collide, they do so elastically: their velocities
        change such that both energy and momentum are conserved.

        """
        connectedComponents = [c for c in sorted(
            nx.connected_components(self.network), key=len, reverse=True)]
        for obj1_pIDs, obj2_pIDs in itertools.combinations(connectedComponents, r=2):
            # print(obj1_pIDs, obj2_pIDs, self.getNetNetContactSite(obj1_pIDs, obj2_pIDs))
            if any(self.getNetNetContactSite(obj1_pIDs, obj2_pIDs)):
                # print("COLLISION")
                self.change_velocities(obj1_pIDs, obj2_pIDs)

    def handle_boundary_collisions(self, pID_list: List[int]):
        """Bounce the particles off the walls elastically.
        Args:
            pID_list (List[int]): list of particles IDs part of the structure
        """

        for pID in pID_list:
            self.particle_dict[pID].checkIsWrappedReflectedAndUpdate(self.sim_boundaries.upper_x,
                                                                     self.sim_boundaries.lower_x,
                                                                     self.sim_boundaries.upper_y,
                                                                     self.sim_boundaries.lower_y)

            self.network.nodes[pID]['obj'] = self.particle_dict[pID]
        return

        ####### Start of reflective boundary condition situation #######
        min_X, max_X = np.inf, -np.inf
        min_Y, max_Y = np.inf, -np.inf

        for pID in pID_list:
            temp_max_X = self.particle_dict[pID].x + \
                self.particle_dict[pID].radius
            temp_max_Y = self.particle_dict[pID].y + \
                self.particle_dict[pID].radius
            temp_min_X = self.particle_dict[pID].x - \
                self.particle_dict[pID].radius
            temp_min_Y = self.particle_dict[pID].y - \
                self.particle_dict[pID].radius

            if temp_min_X < min_X:
                min_X = temp_min_X
            if temp_min_Y < min_Y:
                min_Y = temp_min_Y
            if temp_max_X > max_X:
                max_X = temp_max_X
            if temp_max_Y > max_Y:
                max_Y = temp_max_Y

        if min_X < self.sim_boundaries.lower_x:
            p_update = np.array([self.sim_boundaries.lower_x - min_X, 0])
            v_update = np.array([-1, 1])
            self.updatePositionsAdditive(pID_list=pID_list, delta_Pxy=p_update)
            self.updateVelocitiesMultiply(
                pID_list=pID_list, alpha_Vxy=v_update)

        if max_X > self.sim_boundaries.upper_x:
            px_update = np.array([self.sim_boundaries.upper_x - max_X, 0])
            v_update = np.array([-1, 1])
            self.updatePositionsAdditive(
                pID_list=pID_list, delta_Pxy=px_update)
            self.updateVelocitiesMultiply(
                pID_list=pID_list, alpha_Vxy=v_update)

        if min_Y < self.sim_boundaries.lower_y:
            p_update = np.array([0, self.sim_boundaries.lower_y - min_Y])
            v_update = np.array([1, -1])
            self.updatePositionsAdditive(pID_list=pID_list, delta_Pxy=p_update)
            self.updateVelocitiesMultiply(
                pID_list=pID_list, alpha_Vxy=v_update)

        if max_Y > self.sim_boundaries.upper_y:
            p_update = np.array([0, self.sim_boundaries.upper_y - max_Y])
            v_update = np.array([1, -1])
            self.updatePositionsAdditive(pID_list=pID_list, delta_Pxy=p_update)
            self.updateVelocitiesMultiply(
                pID_list=pID_list, alpha_Vxy=v_update)

        ####### End of reflective boundary condition situation #######

    def updateHistoryMap(self):
        """Update the history map of the simulation
        """
        connectedComponents = [c for c in sorted(
            nx.connected_components(self.network), key=len, reverse=True)]
        nodes2Add = []
        edges2Add = []
        recent_IDs = [v for v, d in self.history.out_degree() if d == 0]
        for subnet_pIDs in connectedComponents:
            subnet_pIDs = list(subnet_pIDs)
            self.historyID += 1
            for rID in recent_IDs:
                # print(rID, subnet_pIDs)
                hData = list(self.history.nodes[rID].get('members'))
                # check = any(pID in hData for pID in subnet_pIDs)
                if not set(hData).isdisjoint(subnet_pIDs):
                    has_agg = self.getSubnetAggStatus(subnet_pIDs)
                    nodes2Add.append(
                        [self.historyID, subnet_pIDs, has_agg, self.timestamp])
                    edges2Add.append((rID, self.historyID))
        for id, membs, status, timestamp in nodes2Add:
            self.history.add_node(
                id, members=membs, has_agg=status, timestamp=timestamp)
        self.history.add_edges_from(edges2Add)

    def advance(self):
        remodeling_occurs = False
        # Position updates
        # Traverse each individual structure
        connectedComponents = [c for c in sorted(
            nx.connected_components(self.network), key=len, reverse=True)]
        for subnet_member_pIDs in connectedComponents:
            subnet_member_pIDs = list(subnet_member_pIDs)
            subnet_aggStatus = self.getSubnetAggStatus(
                ID_list=subnet_member_pIDs)
            # TODO Change this heuristic if particles are allowed to be different areas
            structure_size = len(subnet_member_pIDs)
            # Get diffusion position update place
            # Pick diffusion function depending on structure status
            selected_diffusion = self.diffusion_Agg_func if subnet_aggStatus else self.diffusion_nonAgg_func
            # calculate position update
            pos_update = selected_diffusion(structure_size=structure_size)

            for pID in subnet_member_pIDs:

                # NOTE ALL OF THE FOUR LINES ARE NEEDED BELOW
                # Updates position of the particle in self.particle_dict
                self.particle_dict[pID].advance(position_update=pos_update,
                                                dt=self.time_interval)
                # Updates the networkx representation of the particle in self.particle dict,
                # updating the particle position and the actual particle object itself.
                self.network.nodes[pID]['pos'] = self.particle_dict[pID].position
                self.network.nodes[pID]['obj'] = self.particle_dict[pID]

            self.handle_boundary_collisions(pID_list=subnet_member_pIDs)
        self.handle_collisions()

        # Handle fission/fusion operations
        updated_pIDs = []
        for subnet_member_pIDs in connectedComponents:
            subnet_member_pIDs = list(subnet_member_pIDs)
            # Check if all subnet member iDS are in updated_pIDs
            # if all of them are in  updated_pIDs, skip.
            # Step prevents double update to structures that were part of a pre-existing
            # fission/fusion operation
            any_updated = any(
                pID in updated_pIDs for pID in subnet_member_pIDs)
            all_updated = all(
                pID in updated_pIDs for pID in subnet_member_pIDs)
            assert any_updated == all_updated, "Updating one node part of a structure must update entire structure"
            if any_updated and all_updated:
                continue
            subnet_aggStatus = self.getSubnetAggStatus(
                ID_list=subnet_member_pIDs)

            structure_size = len(subnet_member_pIDs)
            selected_event_func = self.p_event_Agg if subnet_aggStatus else self.p_event_nonAgg
            event = selected_event_func(structure_size=int(structure_size))
            if event == "fus":
                # print("fus")
                # TODO if undergoing fusion, find closest neighboring structure
                fusion_neighbors = self.getNetworkNeighbors(
                    list_pIDs=subnet_member_pIDs)
                # If no such neighbors are within the defined range, add pIDs to visited list
                # so that there isn't a double traversal.
                if not fusion_neighbors:
                    updated_pIDs += subnet_member_pIDs
                    continue
                
                # grab interacting nodeIDs as well as the positions used to measure
                # the interacting distance. Note grabs the first element of the list of 
                # possible fusion neighbors, even though the self.getNetworkNeighbors
                # provides the whole list of neighbors who are valid for interacting with
                selected_neighborPID, selected_neighbor_posType = fusion_neighbors[0][-1]
                selected_selfPID, selected_self_posType = fusion_neighbors[0][1]

                # TODO get vector to snap other structure to current structure
                pos_difference = self.particle_dict.get(selected_neighborPID).getPositionByName(selected_neighbor_posType) - \
                                 self.particle_dict.get(selected_selfPID).getPositionByName(selected_self_posType)

                radius_sum = (self.particle_dict.get(selected_neighborPID).radius +
                              self.particle_dict.get(selected_selfPID).radius)

                separation = np.linalg.norm(pos_difference)
                # Calculate position update needed for fusion
                # make into unit vector
                axis_vector = pos_difference / separation
                axis_vector *= (radius_sum - separation)
                
                # TODO Update all positions of all partner nodes that are undergoing fusion
                other_structure_pIDs = self.getConnectedNodes(
                    selected_neighborPID)
                self.updatePositionsAdditive(other_structure_pIDs,
                                             delta_Pxy=axis_vector)
                # TODO update all partner velocities
                # self.replaceVelocities(other_structure_pIDs, new_Vxy=axis_vector/self.time_interval)
                # TODO form edge with the nearby node
                self.network.add_edge(selected_neighborPID,
                                      selected_selfPID,
                                      sep=separation)
                # TODO add all elements of the query structure and its fused partner to the updated list
                updated_pIDs += other_structure_pIDs
                self.handle_boundary_collisions(
                    pID_list=subnet_member_pIDs + other_structure_pIDs)
                self.handle_collisions()
                remodeling_occurs = True
                # Create copy of network state in history dictionary, with key being timestamp
                # self.network_history[f"{self.timestamp}_fus{event_number:02d}_b"] = deepcopy(
                #     self.network)
                # event_number += 1
            elif event == "fis":
                
                # print("fis")
                # TODO Isolate structure
                subgraph = self.network.subgraph(subnet_member_pIDs)
                subgraph_edges = list(subgraph.edges())
                # If there are noedges to cut, continue script
                # Redundant catch if ProbOperationVol fails
                if len(subgraph_edges) == 0:
                    continue
                # TODO if aggregate harboring, and targeted_fis is set to true, pick edge closest to aggregate.
                # If targeted fission percentage of time is greater than 0
                if self.targeted_fis_probability > 0.0 and subnet_aggStatus:
                    
                    
                    targeted_fis_choice = np.random.choice(a=[True, False],
                                                           p=[self.targeted_fis_probability, 
                                                              1-self.targeted_fis_probability])
                    if targeted_fis_choice:
                        for edge_src_ID, edge_dst_ID in subgraph_edges:
                            src_status = self.network.nodes[edge_src_ID]['obj'].has_agg
                            dst_status = self.network.nodes[edge_dst_ID]['obj'].has_agg
                            # If both nodes have an aggregate
                            if src_status and dst_status:
                                # if aggs are indivisible, skip, otherwise include
                                if self.indivisible_aggs:
                                    pass
                                else:
                                    selected_edge = (edge_src_ID, edge_dst_ID)
                                    break
                            # if only one node has agg, use that edge.
                            elif src_status ^ dst_status:
                                selected_edge = (edge_src_ID, edge_dst_ID)
                                break
                            # if neither node has agg
                            else:
                                pass
                    else:
                        # Pick random edge
                        selected_edge = subgraph_edges[np.random.choice(len(subgraph_edges))]
                        # Run check for indivisible aggs. Will execute regardless 
                        # unless indivisible_aggs setting is set to True
                        if self.indivisible_aggs and self.network.nodes[selected_edge[0]]['obj'].has_agg and self.network.nodes[selected_edge[-1]]['obj'].has_agg:
                            selected_edge = (None, None)
                # TODO if non-targeted fission, just pick random edge.
                else:
                    # Pick random edge
                    selected_edge = subgraph_edges[np.random.choice(len(subgraph_edges))]
                    # Run check for indivisible aggs. Will execute regardless 
                    # unless indivisible_aggs setting is set to True
                    if self.indivisible_aggs and self.network.nodes[selected_edge[0]]['obj'].has_agg and self.network.nodes[selected_edge[-1]]['obj'].has_agg:
                        selected_edge = (None, None)

                # Break edge
                if all([isinstance(item, int) for item in selected_edge]):
                    self.network.remove_edge(selected_edge[0], 
                                             selected_edge[-1])
                    # repel the src_obj and dest_obj resulting from fission 
                    src_pID, dest_pID = selected_edge
                    src_pos_assignments = self.particle_dict.get(src_pID).getAllPositions()
                    dest_pos_assignments = self.particle_dict.get(dest_pID).getAllPositions()
                    # Get minimum distance separating src and dst particle
                    
                    # Returned tuple contains position type in first column, actual position in second column
                    # separate into separate variables
                    src_pos_types_only = list(zip(*src_pos_assignments))[0]
                    src_pos_only = list(zip(*src_pos_assignments))[-1]
                    dest_pos_types_only = list(zip(*dest_pos_assignments))[0]
                    dest_pos_only = list(zip(*dest_pos_assignments))[-1]

                    # compute the distances between sets of points for query and other particle
                    distance_mx = cdist(src_pos_only, dest_pos_only)
                    # Determine minimum separation
                    min_dist = np.min(distance_mx)
                    src_minpos_type_idx, dest_minpos_type_idx = np.argwhere(
                        distance_mx == min_dist)[0]

                    src_minpos_type = src_pos_types_only[src_minpos_type_idx]
                    dest_minpos_type = dest_pos_types_only[dest_minpos_type_idx]
                    
                    fis_axis_vector = self.particle_dict.get(src_pID).getPositionByName(src_minpos_type) - \
                                      self.particle_dict.get(dest_pID).getPositionByName(dest_minpos_type)
                                      
                    fis_axis_vector_mag = np.linalg.norm(fis_axis_vector)
                    # Scale fission vector to repulsion 
                    # article : Mitochondrial membrane tension governs fission
                    fis_axis_vector /= fis_axis_vector_mag
                    fis_axis_vector *= FISSION_EPSILON/2

                    src_obj_pIDs = self.getConnectedNodes(query_pID=src_pID)
                    dest_obj_pIDs = self.getConnectedNodes(query_pID=dest_pID)


                    self.updatePositionsAdditive(src_obj_pIDs, delta_Pxy=fis_axis_vector)
                    self.updatePositionsAdditive(dest_obj_pIDs, delta_Pxy=-fis_axis_vector)
                    # TODO Handle collisions and boundary collisions
                    self.handle_boundary_collisions(src_obj_pIDs)
                    self.handle_boundary_collisions(dest_obj_pIDs)
                    self.handle_collisions()
                    remodeling_occurs = True
            else:
                # TODO if idle, do nothing
                pass
            updated_pIDs += subnet_member_pIDs

        self.nSteps += 1
        self.timestamp += self.time_interval
        self.updateHistoryMap()
        if remodeling_occurs:
            self.network_history[self.timestamp] = deepcopy(self.network)
        self.nAgg_history += [self.count_nAggClusters()]

    def runNIter(self, nIter: int, print_progress: bool = False) -> None:
        """Run the simulation for nIterations

        Args:
            nIter (int): number of iterations to advance
        """
        t0 = time.time()
        if print_progress:
            printProgressBar(0, nIter-1)
        for steps in range(nIter):
            self.advance()
            if print_progress:
                printProgressBar(steps, int(nIter)-1)
        t_end = time.time()
        print(f"\nRuntime of : {t_end-t0}s")

    def runUntilTimestamp(self, stopTime: int, print_progress: bool = False) -> None:
        """Run simulation until stopTime (in seconds), unless stopTime is 
        less than that of the present time recorded in the simulation.

        Args:
            stopTime (int): timestamp to stop at

        Raises:
            Exception: If self.timestamp is greater than that of stoptime, 
                cannot run simulation without jumping time

        """

        if stopTime <= self.timestamp:
            raise Exception(
                f"Simulation has already been run until {self.timestamp}s")
        # Progress bar for printing progress of simulation

        t0 = time.time()
        if print_progress:
            nIter = int((stopTime - self.timestamp)/self.time_interval)
            printProgressBar(0, nIter-1)
            step_num = 0
        while self.timestamp < stopTime:
            self.advance()
            if print_progress:
                printProgressBar(step_num, nIter-1)
                step_num += 1
        t_end = time.time()
        print(f"\nRuntime of : {t_end-t0}s")

    def drawReducedHistory(self) -> None:
        """Plot a reduced history plot for fission/fusion events in the simulation.
        """
        outputGraph = simplifyDG(self.history)
        pos = nx.nx_agraph.graphviz_layout(outputGraph,
                                prog='dot',
                                args="-Grankdir=LR")
        node_color = [outputGraph.nodes[node]['timestamp']
                      for node in outputGraph.nodes()]
        node_sizes = [len(outputGraph.nodes[node]['members'])
                      for node in outputGraph.nodes()]
        cmap = plt.cm.Reds
        nx.draw_networkx(outputGraph,
                         pos=pos,
                         node_color=node_color,
                         node_size = node_sizes,
                         cmap=cmap,
                         with_labels=False,
                        #  node_size=5,
                         connectionstyle="angle, angleA=-90,angleB=180, rad=0")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min(node_color),
                                                                 vmax=max(node_color)))
        sm._A = []
        cbar = plt.colorbar(sm)
        cbar.set_label("timestamp")
        plt.title(f"Compressed history map at t={self.timestamp:02f}s")
        plt.show()
        return outputGraph

    def drawSelf(self, show=True) -> None:
        """Draw the particles and network for the simulation at its current state.
        """
        _, ax = plt.subplots()
        ax.set_xlim(self.sim_boundaries.lower_x-self.default_particle_diameter*2,
                    self.sim_boundaries.upper_x+self.default_particle_diameter*2)
        ax.set_ylim(self.sim_boundaries.lower_y-self.default_particle_diameter*2,
                    self.sim_boundaries.upper_y+self.default_particle_diameter*2)

        ax.add_patch(Rectangle([0, 0], self.sim_boundaries.width,
                     self.sim_boundaries.height, fill=False, linestyle=":"))
        ax.add_patch(Rectangle([-self.default_particle_diameter, -self.default_particle_diameter],
                               self.sim_boundaries.width+2*self.default_particle_diameter,
                               self.sim_boundaries.height+2*self.default_particle_diameter,
                               fill=False, linestyle=":"))

        bottom_box = Rectangle([-self.default_particle_diameter, -self.default_particle_diameter],
                               self.sim_boundaries.width + 2 * self.default_particle_diameter,
                               self.default_particle_diameter,
                               color=GREY_COLOR, alpha=SPINE_ALPHA, fill=True)
        top_box = Rectangle([-self.default_particle_diameter,
                             self.sim_boundaries.height],
                            self.sim_boundaries.width + 2 * self.default_particle_diameter,
                            self.default_particle_diameter,
                            color=GREY_COLOR, alpha=SPINE_ALPHA, fill=True)
        left_box = Rectangle([-self.default_particle_diameter,
                             -self.default_particle_diameter],
                             self.default_particle_diameter,
                             self.sim_boundaries.width + 2 * self.default_particle_diameter,
                             color=GREY_COLOR, alpha=SPINE_ALPHA, fill=True)
        right_box = Rectangle([self.sim_boundaries.width,
                               -self.default_particle_diameter],
                              self.default_particle_diameter,
                              self.sim_boundaries.width + 2 * self.default_particle_diameter,
                              color=GREY_COLOR, alpha=SPINE_ALPHA, fill=True)

        ax.add_patch(bottom_box)
        ax.add_patch(top_box)
        ax.add_patch(left_box)
        ax.add_patch(right_box)

        for _, nodeData in self.network.nodes(data=True):
            nodeData['obj'].drawSelf(ax)
            # add original object
            ax.add_patch(Circle(xy=nodeData['obj'].position,
                                radius=nodeData['obj'].radius,
                                **nodeData['obj'].drawstyle))
            ax.text(x=nodeData['obj'].x,
                    y=nodeData['obj'].y,
                    color=TEXT_COLOR,
                    fontsize=PARTICLE_LABEL_SIZE,
                    s=f"{nodeData['obj'].ID}",
                    horizontalalignment="center",
                    verticalalignment="center")

            if nodeData['obj'].is_wrapped_around:
                ax.add_patch(Circle(xy=nodeData['obj'].wrapped_pos,
                                    radius=nodeData['obj'].radius,
                                    **nodeData['obj'].drawstyle))
                ax.text(x=nodeData['obj'].wrap_x,
                        y=nodeData['obj'].wrap_y,
                        color=TEXT_COLOR,
                        fontsize=PARTICLE_LABEL_SIZE,
                        s=f"{nodeData['obj'].ID}w",
                        horizontalalignment="center",
                        verticalalignment="center")
            if nodeData['obj'].has_reflections:
                for reflect_pos in nodeData['obj'].reflect_list:
                    if reflect_pos is None:
                        continue
                    else:
                        temp_circle_patch = Circle(xy=reflect_pos,
                                                   radius=nodeData['obj'].radius,
                                                   **nodeData['obj'].drawstyle)
                    # reflected patches are labeled with a R
                    ax.text(x=reflect_pos[0],
                            y=reflect_pos[1],
                            s=f"{nodeData['obj'].ID}R",
                            color=TEXT_COLOR,
                            fontsize=PARTICLE_LABEL_SIZE,
                            horizontalalignment="center",
                            verticalalignment="center")
                    ax.add_patch(temp_circle_patch)

        for edge_src, edge_dst in self.network.edges():
            src_x, src_y = self.network.nodes[edge_src]['obj'].position
            dst_x, dst_y = self.network.nodes[edge_dst]['obj'].position
            ax.plot([src_x, dst_x], [src_y, dst_y],
                    marker="o", markersize=2,
                    color=GREY_COLOR,
                    linewidth=LINEWIDTH)
        plt.gca().set_aspect('equal', adjustable='box')
        if show:
            plt.show()

    def setup_animation(self) -> None:
        """sets up animation axes for plotting
        """
        self.fig, self.ax = plt.subplots()
        for s in ['top', 'bottom', 'left', 'right']:
            self.ax.spines[s].set_linewidth(2)

        # Draw boundary around simulation box
        self.ax.set_xlim(self.sim_boundaries.lower_x-self.default_particle_diameter*2,
                         self.sim_boundaries.upper_x+self.default_particle_diameter*2)
        self.ax.set_ylim(self.sim_boundaries.lower_y-self.default_particle_diameter*2,
                         self.sim_boundaries.upper_y+self.default_particle_diameter*2)
        # draw simulation box region
        sim_box_artist = Rectangle([0, 0],
                                   self.sim_boundaries.width, self.sim_boundaries.height,
                                   fill=False, linestyle=":")
        self.ax.add_artist(sim_box_artist)
        sim_box_artist = Rectangle([-self.default_particle_diameter, -self.default_particle_diameter],
                                   self.sim_boundaries.width+2*self.default_particle_diameter,
                                   self.sim_boundaries.height+2*self.default_particle_diameter,
                                   fill=False, linestyle=":")
        self.ax.add_artist(sim_box_artist)
        bottom_box = Rectangle([-self.default_particle_diameter, -self.default_particle_diameter],
                               self.sim_boundaries.width + 2 * self.default_particle_diameter,
                               self.default_particle_diameter,
                               color=GREY_COLOR, alpha=SPINE_ALPHA, fill=True)
        top_box = Rectangle([-self.default_particle_diameter,
                             self.sim_boundaries.height],
                            self.sim_boundaries.width + 2 * self.default_particle_diameter,
                            self.default_particle_diameter,
                            color=GREY_COLOR, alpha=SPINE_ALPHA, fill=True)
        left_box = Rectangle([-self.default_particle_diameter,
                             -self.default_particle_diameter],
                             self.default_particle_diameter,
                             self.sim_boundaries.width + 2 * self.default_particle_diameter,
                             color=GREY_COLOR, alpha=SPINE_ALPHA, fill=True)
        right_box = Rectangle([self.sim_boundaries.width,
                               -self.default_particle_diameter],
                              self.default_particle_diameter,
                              self.sim_boundaries.width + 2 * self.default_particle_diameter,
                              color=GREY_COLOR, alpha=SPINE_ALPHA, fill=True)
        self.ax.add_artist(bottom_box)
        self.ax.add_artist(left_box)
        self.ax.add_artist(right_box)
        self.ax.add_artist(top_box)
        self.ax.set_aspect('equal', 'box')

    def advance_animation(self) -> List[matplotlib.patches.Circle]:
        """Generates a list of artists from the simulation for real-time rendering
        of the simulation

        Returns:
            List[matplotlib.artist]: list of matplotlib artists
        """
        self.advance()
        # get artists for drawing
        self.circles = []
        for particle_obj in self.particle_dict.values():
            self.circles.append(particle_obj.drawSelf(self.ax))

            # add original object
            temp_circle_patch = Circle(xy=particle_obj.position,
                                       radius=particle_obj.radius,
                                       **particle_obj.drawstyle)
            txt_lab = self.ax.text(x=particle_obj.x,
                                   y=particle_obj.y,
                                   s=f"{particle_obj.ID}",
                                   color=TEXT_COLOR,
                                   fontsize=PARTICLE_LABEL_SIZE,
                                   horizontalalignment="center",
                                   verticalalignment="center")
            self.ax.add_patch(temp_circle_patch)
            self.circles.append(temp_circle_patch)
            self.circles.append(txt_lab)

            # If particle is on boundary, add additional patch for visualization
            if particle_obj.is_wrapped_around:
                temp_circle_patch = Circle(xy=particle_obj.wrapped_pos,
                                           radius=particle_obj.radius,
                                           **particle_obj.drawstyle)
                # wrapped patches are labeled with a W
                txt_lab = self.ax.text(x=particle_obj.wrap_x,
                                       y=particle_obj.wrap_y,
                                       s=f"{particle_obj.ID}W",
                                       color=TEXT_COLOR,
                                       fontsize=PARTICLE_LABEL_SIZE,
                                       horizontalalignment="center",
                                       verticalalignment="center")
                self.ax.add_patch(temp_circle_patch)
                self.circles.append(temp_circle_patch)
                self.circles.append(txt_lab)

            if particle_obj.has_reflections:
                for reflect_pos in particle_obj.reflect_list:
                    if reflect_pos is None:
                        continue
                    else:
                        temp_circle_patch = Circle(xy=reflect_pos,
                                                   radius=particle_obj.radius,
                                                   **particle_obj.drawstyle)
                    # reflected patches are labeled with a R
                    txt_lab = self.ax.text(x=reflect_pos[0],
                                           y=reflect_pos[1],
                                           s=f"{particle_obj.ID}R",
                                           color=TEXT_COLOR,
                                           fontsize=PARTICLE_LABEL_SIZE,
                                           horizontalalignment="center",
                                           verticalalignment="center")
                    self.ax.add_patch(temp_circle_patch)
                    self.circles.append(temp_circle_patch)
                    self.circles.append(txt_lab)
            else:
                pass

        for edge_src, edge_dst in self.network.edges():
            src_x, src_y = self.network.nodes[edge_src]['obj'].position
            dst_x, dst_y = self.network.nodes[edge_dst]['obj'].position
            b = self.ax.plot([src_x, dst_x], [src_y, dst_y],
                             marker="o", markersize=2,
                             color=GREY_COLOR,
                             linewidth=LINEWIDTH)
            self.circles.append(b[0])
        return self.circles

    def animate(self, i: int) -> List[matplotlib.patches.Circle]:
        """Advances the animation of the simulation

        Args:
            i (int): placeholder variable for FuncAnimation

        Returns:
            List[matplotlib.artist]: list of matplotlib artists
        """
        self.advance_animation()
        # self.ax.cla()
        return self.circles

    def init_animation(self) -> List[matplotlib.patches.Circle]:
        """Generate artists for initial frame of animation

        Returns:
            List[matplotlib.artist]: list of circles/lines used to draw simulation state.
        """
        self.circles = []
        for particle in self.particle_dict.values():
            self.circles.append(particle.drawSelf(self.ax))
        for edge_src, edge_dst in self.network.edges():
            src_x, src_y = self.network.nodes[edge_src]['obj'].position
            dst_x, dst_y = self.network.nodes[edge_dst]['obj'].position
            b = self.ax.plot([src_x, dst_x], [src_y, dst_y],
                             marker="o", markersize=2,
                             color=BLUE_COLOR,
                             linewidth=LINEWIDTH)
            self.circles.append(b[0])
        return self.circles

    def save_or_show_animation(self, anim,
                               save: bool,
                               filename: str = 'simulation.mp4') -> None:
        """helper function for determining whether to save or show animation

        Args:
            anim (matplotlib.animation.FuncAnimation): animation object
            save (bool): Set to true to save to file
            filename (str, optional): Filepath to save file to. Defaults to 'simulation.mp4'.
        """
        if save:
            Writer = matplotlib.animation.writers['ffmpeg']
            writer = Writer(fps=30, bitrate=1800)
            anim.save(filename, writer=writer, dpi=900)
        else:
            plt.show()

    def do_animation(self, save: bool = False, filename: str = 'simulation.mp4'):
        """Set up and carry out the animation of the molecular dynamics.

        To save the animation as a MP4 movie, set save=True.
        """

        self.setup_animation()
        anim = matplotlib.animation.FuncAnimation(self.fig, self.animate,
                                                  init_func=self.init_animation,
                                                  frames=100,
                                                  interval=self.time_interval*1000, blit=True)
        self.save_or_show_animation(anim, save, filename)


