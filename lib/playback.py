import numpy as np
import pickle
import networkx as nx
import matplotlib.pyplot as plt
from typing import List
from matplotlib.patches import Rectangle, Circle
from lib.simulation_handler import Simulation
from configs.graphics_config import (
    BLUE_COLOR, GREY_COLOR, LINEWIDTH, TEXT_COLOR, PARTICLE_LABEL_SIZE, SPINE_ALPHA)


class PlaybackSimulation:
    def __init__(self, simulation_pickle_path:str) -> None:
        """Class for playback and visualization of simulation over time.
        Only plays frames where a change in structure is observed.

        Args:
            simulation_pickle_path (str): Path to serialized simulation pickle object
        """
        self.simulation_pickle_path = simulation_pickle_path
        with open(self.simulation_pickle_path, 'rb') as f:
            self.simulation_obj = pickle.load(f)

        self.timestamps = list(self.simulation_obj.network_history.keys())

    def playthrough(self, start_time: float, end_time: float, time_resolution: float) -> None:
        """Playthrough function for simulation.

        Args:
            start_time (float): timestamp in simulation to start playing at
            end_time (float): timestamp in simulation to stop playback
            time_resolution (float): time resolution to play at. Because the simulation does not record every frame, 
                sometimes this may skip times larger than time_resolution.  When set to -1, plays every frame.
        """
        assert start_time <= max(
            self.timestamps), "start time must be less than whole simulation duration"
        assert end_time <= max(
            self.timestamps), "end time must be less than whole simulation duration"
        assert time_resolution <= (
            end_time - start_time), "time resolution must be smaller than time duration to playthrough"
        assert end_time > start_time, "end time must be greater than start time"
        for timestamp, network_state in self.simulation_obj.network_history.items():
            if timestamp <= start_time:
                continue
            if timestamp > end_time:
                break
            if time_resolution == -1:
                pass
            else:
                if (np.rint(timestamp) - start_time) % time_resolution != 0:
                    continue
            plt.figure(figsize=(5, 5))
            ax = plt.gca()
            ax.set_title(f'Time: {timestamp} (s)')
            pos = [p['pos'] for _, p in network_state.nodes(data=True)]
            c = ["g" if not p['has_agg'] else "r" for _,
                 p in network_state.nodes(data=True)]
            nx.draw(network_state,
                    with_labels=True,
                    pos=pos, node_color=c, ax=ax)
            _ = ax.axis('off')
            plt.show()

    def drawAllFrames(self) -> None:
        """Draws all of the grames of the simulation over time, draws only frames in which there have
        been network changes since the last state.  Does not draw every frame where particles
        have only moved but not undergone fission/fusion
        Saves each individual frame to a png file.
        """
        for timestamp_ID, network_state in self.simulation_obj.network_history.items():
            plt.figure(figsize=(9, 9))
            ax = plt.gca()
            approx_timestamp = np.around(timestamp_ID,2)
            ax.set_title(f'Key: {approx_timestamp:.2f}')

            ax.set_xlim(self.simulation_obj.sim_boundaries.lower_x-self.simulation_obj.default_particle_diameter*2,
                        self.simulation_obj.sim_boundaries.upper_x+self.simulation_obj.default_particle_diameter*2)
            ax.set_ylim(self.simulation_obj.sim_boundaries.lower_y-self.simulation_obj.default_particle_diameter*2,
                        self.simulation_obj.sim_boundaries.upper_y+self.simulation_obj.default_particle_diameter*2)
            # set simulation box
            ax.add_patch(Rectangle([0, 0], self.simulation_obj.sim_boundaries.width,
                                   self.simulation_obj.sim_boundaries.height, fill=False, linestyle=":"))
            # Set outside box
            ax.add_patch(Rectangle([-self.simulation_obj.default_particle_diameter, -self.simulation_obj.default_particle_diameter],
                                   self.simulation_obj.sim_boundaries.width+2*self.simulation_obj.default_particle_diameter,
                                   self.simulation_obj.sim_boundaries.height+2*self.simulation_obj.default_particle_diameter,
                                   fill=False, linestyle=":"))
            
            bottom_box = Rectangle([-self.simulation_obj.default_particle_diameter, -self.simulation_obj.default_particle_diameter],
                               self.simulation_obj.sim_boundaries.width + 2 * self.simulation_obj.default_particle_diameter,
                               self.simulation_obj.default_particle_diameter,
                               color=GREY_COLOR, alpha=SPINE_ALPHA, fill=True)
            top_box = Rectangle([-self.simulation_obj.default_particle_diameter,
                                self.simulation_obj.sim_boundaries.height],
                                self.simulation_obj.sim_boundaries.width + 2 * self.simulation_obj.default_particle_diameter,
                                self.simulation_obj.default_particle_diameter,
                                color=GREY_COLOR, alpha=SPINE_ALPHA, fill=True)
            left_box = Rectangle([-self.simulation_obj.default_particle_diameter,
                                -self.simulation_obj.default_particle_diameter],
                                self.simulation_obj.default_particle_diameter,
                                self.simulation_obj.sim_boundaries.width + 2 * self.simulation_obj.default_particle_diameter,
                                color=GREY_COLOR, alpha=SPINE_ALPHA, fill=True)
            right_box = Rectangle([self.simulation_obj.sim_boundaries.width,
                                -self.simulation_obj.default_particle_diameter],
                                self.simulation_obj.default_particle_diameter,
                                self.simulation_obj.sim_boundaries.width + 2 * self.simulation_obj.default_particle_diameter,
                                color=GREY_COLOR, alpha=SPINE_ALPHA, fill=True)
            ax.add_patch(bottom_box)
            ax.add_patch(top_box)
            ax.add_patch(left_box)
            ax.add_patch(right_box)
           
            for node, nodeData in network_state.nodes(data=True):
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
            for edge_src, edge_dst in network_state.edges():
                src_x, src_y = network_state.nodes[edge_src]['obj'].position
                dst_x, dst_y = network_state.nodes[edge_dst]['obj'].position
                ax.plot([src_x, dst_x], [src_y, dst_y],
                        marker="o", markersize=2,
                        color=GREY_COLOR,
                        linewidth=LINEWIDTH)
            _ = ax.axis('off')
            ax.set_aspect('equal', adjustable='box')
            plt.savefig(f"frame_t_{approx_timestamp}.png")
            plt.close()

    def drawEndState(self):
        """Draws the end state of the simulation"""
        self.simulation_obj.drawSelf()

    def plotHistory(self):
        """Draws the end state of the simulation and the simplified
        fission/fusion history network of the simulation
        """
        self.simulation_obj.drawSelf()
        self.simulation_obj.drawReducedHistory()

    def printTimestamps(self):
        """Prints the timestamps for which the simulation has recorded the simulation state"""
        for timestamp in self.simulation_obj.network_history.keys():
            print(timestamp)

        
    def playthroughTargFis(self, depth: int = 3)-> None:
        """Generate frames for simulation events where a reduction in number of aggregate clusters is observed.
        Used for generating images of simulation state and observing what happens when the number of aggregate
        clusters changes and what processes drive this process.
        
        Since simulation states are only recorded when the simulation has had a change in the network topology, 
        the depth parameter controls the number of frames forwards and backwards in which there has been a 
        change in the network caused by fission/fusion.  The resulting time-samples sampled may not be equally 
        spaced.
        
        If there are no aggregate reduction events, nothing will be plotted.

        Args:
            depth (int, optional): Number of frames forwards and backwards to capture around the cluster 
                number change event. Defaults to 3.
        """
        pre_idx_decrease = []
        for k, (i,j) in enumerate(zip(self.simulation_obj.nAgg_history[0::1], self.simulation_obj.nAgg_history[1::1])):
            if j<i:
                pre_idx_decrease += [k]
                
        for pre_idx in pre_idx_decrease:
            timestamp = self.simulation_obj.time_interval * pre_idx
            
            pre_timestamps = [i for i in self.simulation_obj.network_history.keys() if i <=timestamp ]
            post_timestamps = [i for i in self.simulation_obj.network_history.keys() if i >timestamp ]
            
            pre_timestamps = pre_timestamps[-depth:]
            post_timestamps = post_timestamps[:depth]
            for t_i in pre_timestamps:
                self.drawSimState(self.simulation_obj.network_history.get(t_i), t_i)
            print("post reduction")
            for t_i in post_timestamps:
                self.drawSimState(self.simulation_obj.network_history.get(t_i), t_i)
                
            
           
    def drawSimState(self, sim_nx_state: nx.Graph, timestamp_ID:float):
        """Given a timestamp ID of the simulation, and the connectivity state of the simulation at that time,
        Draw the simulation state at that timestamp

        Args:
            sim_nx_state (nx.Graph): networkx undirected graph mapping out particle connections
            timestamp_ID (float): timestamp of simulation frame desired
        """
        plt.figure(figsize=(5, 5))
        ax = plt.gca()
        approx_timestamp = np.around(timestamp_ID,2)
        ax.set_title(f'Key: {approx_timestamp:.2f}')

        ax.set_xlim(self.simulation_obj.sim_boundaries.lower_x-self.simulation_obj.default_particle_diameter*2,
                    self.simulation_obj.sim_boundaries.upper_x+self.simulation_obj.default_particle_diameter*2)
        ax.set_ylim(self.simulation_obj.sim_boundaries.lower_y-self.simulation_obj.default_particle_diameter*2,
                    self.simulation_obj.sim_boundaries.upper_y+self.simulation_obj.default_particle_diameter*2)
        # set simulation box
        ax.add_patch(Rectangle([0, 0], self.simulation_obj.sim_boundaries.width,
                                self.simulation_obj.sim_boundaries.height, fill=False, linestyle=":"))
        # Set outside box
        ax.add_patch(Rectangle([-self.simulation_obj.default_particle_diameter, -self.simulation_obj.default_particle_diameter],
                                self.simulation_obj.sim_boundaries.width+2*self.simulation_obj.default_particle_diameter,
                                self.simulation_obj.sim_boundaries.height+2*self.simulation_obj.default_particle_diameter,
                                fill=False, linestyle=":"))
        
        bottom_box = Rectangle([-self.simulation_obj.default_particle_diameter, -self.simulation_obj.default_particle_diameter],
                            self.simulation_obj.sim_boundaries.width + 2 * self.simulation_obj.default_particle_diameter,
                            self.simulation_obj.default_particle_diameter,
                            color=GREY_COLOR, alpha=SPINE_ALPHA, fill=True)
        top_box = Rectangle([-self.simulation_obj.default_particle_diameter,
                            self.simulation_obj.sim_boundaries.height],
                            self.simulation_obj.sim_boundaries.width + 2 * self.simulation_obj.default_particle_diameter,
                            self.simulation_obj.default_particle_diameter,
                            color=GREY_COLOR, alpha=SPINE_ALPHA, fill=True)
        left_box = Rectangle([-self.simulation_obj.default_particle_diameter,
                            -self.simulation_obj.default_particle_diameter],
                            self.simulation_obj.default_particle_diameter,
                            self.simulation_obj.sim_boundaries.width + 2 * self.simulation_obj.default_particle_diameter,
                            color=GREY_COLOR, alpha=SPINE_ALPHA, fill=True)
        right_box = Rectangle([self.simulation_obj.sim_boundaries.width,
                            -self.simulation_obj.default_particle_diameter],
                            self.simulation_obj.default_particle_diameter,
                            self.simulation_obj.sim_boundaries.width + 2 * self.simulation_obj.default_particle_diameter,
                            color=GREY_COLOR, alpha=SPINE_ALPHA, fill=True)
        ax.add_patch(bottom_box)
        ax.add_patch(top_box)
        ax.add_patch(left_box)
        ax.add_patch(right_box)
        
        for node, nodeData in sim_nx_state.nodes(data=True):
            nodeData['obj'].drawstyle['alpha'] = 0.5
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
        # for edge_src, edge_dst in sim_nx_state.edges():
        #     src_x, src_y = sim_nx_state.nodes[edge_src]['obj'].position
        #     dst_x, dst_y = sim_nx_state.nodes[edge_dst]['obj'].position
        #     ax.plot([src_x, dst_x], [src_y, dst_y],
        #             marker="o", markersize=2,
        #             color=GREY_COLOR,
        #             linewidth=LINEWIDTH)
        _ = ax.axis('off')
        ax.set_aspect('equal', adjustable='box')
        plt.show()
        


