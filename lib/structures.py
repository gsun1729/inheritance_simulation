import numpy as np
import random
from matplotlib.patches import Circle
from copy import deepcopy
from typing import Tuple, List

from configs.structural_config import DIFFERENCE_EPSILON
from configs.graphics_config import (ALPHA_LEVEL, GREEN_COLOR, RED_COLOR)


class Particle:
    def __init__(self, pos_vec: np.ndarray,
                 vel_vec: np.ndarray,
                 radius: float,
                 ID: int,
                 has_agg: bool):
        """Particle class for mitochondria particles

        Args:
            pos_vec (np.ndarray): 2D position of particle (x,y)
            vel_vec (np.ndarray): 2D velocity vector of particle (dx,dy)
            radius (float): radius of particle in um
            ID (int): Integer ID for particle for tracking
            has_agg (bool): Whether the particle contains an aggregate
        """
        self.position = pos_vec
        self.radius = float(radius)
        self.velocity = vel_vec
        self.has_agg = has_agg
        self.ID = ID
        self.mass = np.pi * (self.radius ** 2)

        # Wrap around condition and position
        self.is_wrapped_around = False
        self.wrapped_pos = np.array([np.inf, np.inf])
        self.has_reflections = False
        self.upper_left_reflect_pos = None
        self.lower_left_reflect_pos = None
        self.upper_right_reflect_pos = None
        self.lower_right_reflect_pos = None

        self.drawstyle = {
            'color': RED_COLOR,
            'fill': True,
            'alpha': ALPHA_LEVEL}
        if self.has_agg:
            self.drawstyle['color'] = GREEN_COLOR

    def __ge__(self, otherParticle):
        return self.mass >= otherParticle.mass

    def __eq__(self, otherParticle):
        return self.mass == otherParticle.mass

    def __le__(self, otherParticle):
        return self.mass <= otherParticle.mass

    @property
    def x(self):
        return self.position[0]

    @x.setter
    def x(self, newX: float):
        self.position[0] = newX

    @property
    def y(self):
        return self.position[-1]

    @y.setter
    def y(self, newY: float):
        self.position[-1] = newY

    @property
    def wrap_x(self):
        return self.wrapped_pos[0]

    @wrap_x.setter
    def wrap_x(self, newX: float):
        self.wrapped_pos[0] = newX

    @property
    def wrap_y(self):
        return self.wrapped_pos[-1]

    @wrap_y.setter
    def wrap_y(self, newY: float):
        self.wrapped_pos[-1] = newY

    @property
    def vx(self):
        return self.velocity[0]

    @vx.setter
    def vx(self, newVx):
        self.velocity[0] = newVx

    @property
    def vy(self):
        return self.velocity[1]

    @vy.setter
    def vy(self, newVy):
        self.velocity[1] = newVy

    @property
    def status(self):
        return self.has_agg

    @status.setter
    def status(self, new_status):
        self.has_agg = new_status
        if self.has_agg:
            self.drawstyle['color'] = GREEN_COLOR
        else:
            self.drawstyle['color'] = RED_COLOR

    def resetWrappedPos(self):
        """Resets the particle as not-wrapped around, used for dealing with periodic boundary condition
        """
        self.is_wrapped_around = False
        self.wrapped_pos = np.array([np.inf, np.inf])
        self.upper_left_wrapped_pos = None
        self.lower_left_wrapped_pos = None
        self.upper_right_wrapped_pos = None
        self.lower_right_wrapped_pos = None

    def swapWrappedPos(self):
        """If the particle is wrapped around, swap the position of the wrapped particle with the original 
        particle.  This function is used to aid in the particle transitioning fully to the other side of the 
        simulation box
        """
        assert self.is_wrapped_around, "particle must be wrapped around to have another position to swap"
        original_pos = deepcopy(self.position)
        original_wrapped_pos = deepcopy(self.wrapped_pos)

        self.position = original_wrapped_pos
        self.wrapped_pos = original_pos

    def getAllPositions(self) -> List[Tuple[str,np.ndarray]]:
        """Get all positions of the particle (including wrapped around positions if valid.)

        Returns:
            List[Tuple[str,np.ndarray]]: List of positions the particle is occupying. 
                tuples are of the form (position_type, position_vector)
        """
        all_positions = [("p", self.position), #registered position
                         ("ul", self.upper_left_reflect_pos), #upper left
                         ("ll", self.lower_left_reflect_pos), #lower left
                         ("ur", self.upper_right_reflect_pos),#upper right
                         ("lr", self.lower_right_reflect_pos)] #lower right
        # need boolean check, since self.wrapped_pos by default is a numpy array
        if self.is_wrapped_around:
            all_positions += [("w", self.wrapped_pos)]#wrapped around position
        all_positions = [(y, x) for (y, x) in all_positions if x is not None]
        return all_positions

    def getPositionByName(self, name: str)-> np.ndarray:
        """Given a particle's position name, get the position of the particle

        Args:
            name (str): position name, has to be one of ("p","ul","ll","ur","lr","w")

        Raises:
            ValueError: If position name is invalid

        Returns:
            np.ndarray: position vector
        """
        if name == "p":
            return self.position
        elif name == "ul":
            return self.upper_left_reflect_pos
        elif name == "ll":
            return self.lower_left_reflect_pos
        elif name == "ur":
            return self.upper_right_reflect_pos
        elif name == "lr":
            return self.lower_right_reflect_pos
        elif name == "w":
            return self.wrapped_pos
        else:
            raise ValueError("unrecognized position name")

    def checkReflectionsExist(self)->bool:
        """Checks to see if the particle has any wrapped around reflections

        Returns:
            bool: True if there is wrapping/reflections
        """
        self.reflect_list = [self.upper_left_reflect_pos,
                             self.lower_left_reflect_pos,
                             self.upper_right_reflect_pos,
                             self.lower_right_reflect_pos]
        if any(elem is not None for elem in self.reflect_list):
            self.has_reflections = True
            self.reflect_list = [x for x in self.reflect_list if x is not None]
            return True
        else:
            self.reflect_list = None
            self.has_reflections = False
            return False

    def checkIsWrappedReflectedAndUpdate(self, sim_upper_x: float, sim_lower_x: float, sim_upper_y: float, sim_lower_y: float):
        """Checks to see if a particle is wrapped around or reflected and updates its position

        Args:
            sim_upper_x (float): simulation box upper x bound
            sim_lower_x (float): simulation box lower x bound
            sim_upper_y (float): simulation box upper y bound
            sim_lower_y (float): simulation box lower y bound

        Raises:
            Exception: Illegal boundary exception, particle can only break 2 walls max
        """

        height = sim_upper_y - sim_lower_y
        width = sim_upper_x - sim_lower_x
        # Check if the particle requires wrapping around
        # If particle is past the boundary of the simulation, wrap around
        # with periodic boundary condition
        particle_is_wrapped = False
        break_left, break_right = False, False
        break_top, break_bottom = False, False

        original_x_pos, original_y_pos = self.position
        # Check if particle boundaries are breaching the boundaries of the simulation, as well as the directionality
        # of that breach

        if self.x + self.radius > sim_upper_x:
            particle_is_wrapped = True
            break_right = True
        if self.x - self.radius < sim_lower_x:
            particle_is_wrapped = True
            break_left = True
        if self.y + self.radius > sim_upper_y:
            particle_is_wrapped = True
            break_top = True
        if self.y - self.radius < sim_lower_y:
            particle_is_wrapped = True
            break_bottom = True
        # Variables for storing reflection positions, these variables will always be none unless
        # the particle boundary breaches more than one boundary (typically when the particle is in the
        # corners of the simulation, where the particle will interact with breaches some permuation of
        # top, bottom vs right, left)
        # notation is as follows:
        # * ul >> particle is in upper left corner
        # * ll >> particle is in lower left corner
        # * ur >> particle is in upper right corner
        # * lr >> particle is in lower right corner
        reflect_ul, reflect_ll, reflect_ur, reflect_lr = None, None, None, None
        # If particle is in the upper right corner
        if (break_right and break_top):
            # reflect to lower left (ll)
            original_x_pos = self.x - width
            original_y_pos = self.y - height
            # Calculate reflections based on ll position
            reflect_ul = np.array([original_x_pos, original_y_pos + height])
            reflect_lr = np.array([original_x_pos + width, original_y_pos])
        # If particle is in the lower right corner
        elif (break_right and break_bottom):
            # reflect to upper left (ul)
            original_x_pos = self.x - width
            original_y_pos = self.y + height
            # Calculate reflections based on ul position
            reflect_ur = np.array([original_x_pos + width, original_y_pos])
            reflect_ll = np.array([original_x_pos, original_y_pos - height])
        # If particle is in the upper left corner
        elif (break_left and break_top):
            # reflect to lower right corner
            original_x_pos = self.x + width
            original_y_pos = self.y - height
            # Calculate reflections based on lr position
            reflect_ur = np.array([original_x_pos, original_y_pos+height])
            reflect_ll = np.array([original_x_pos-width, original_y_pos])
        # If particle is in the lower left corner
        elif (break_left and break_bottom):
            # Reflect to upper right corner
            original_x_pos = self.x + width
            original_y_pos = self.y + height
            # Calculate reflections based on ur position
            reflect_ul = np.array([original_x_pos-width, original_y_pos])
            reflect_lr = np.array([original_x_pos, original_y_pos-height])
        # If particle is breaking only the right wall
        elif (break_right and not break_top and not break_bottom):
            original_x_pos = self.x - width
        # If particle is breaking only the left wall
        elif (break_left and not break_top and not break_bottom):
            original_x_pos = self.x + width
        # If particle is breaking only the top wall
        elif (break_top and not break_right and not break_left):
            original_y_pos = self.y - height
        # If particle is breaking only the bottom wall
        elif (break_bottom and not break_right and not break_left):
            original_y_pos = self.y + height
        elif (not break_top and not break_bottom and not break_right and not break_left):
            # If particle is in the center of the simulation box, do nothing
            pass
        else:
            raise Exception(
                "Illegal boundary exception, particle can only break 2 walls max")
        # If the center of the particle exceeds the boundary of the simulation bounds, particle is looped
        particle_looped = False
        if self.x < sim_lower_x or self.x > sim_upper_x:
            particle_looped = True
        if self.y < sim_lower_y or self.y > sim_upper_y:
            particle_looped = True
            
        # Store reflections (if any)
        self.upper_left_reflect_pos = reflect_ul
        self.lower_left_reflect_pos = reflect_ll
        self.upper_right_reflect_pos = reflect_ur
        self.lower_right_reflect_pos = reflect_lr
        # Update reflections list
        self.checkReflectionsExist()
        # if the particle is wrapped around (and should be present in multiple places)
        if particle_is_wrapped:
            # update particle position
            self.is_wrapped_around = True
            self.wrap_x = original_x_pos
            self.wrap_y = original_y_pos
            # if particle center is past the boundary, the wrapped position will be within the simulation boundary
            # swap positions of wrapped position and primary position
            if particle_looped:
                self.swapWrappedPos()
        else:
            # If particle is not wrapped, reset position
            self.resetWrappedPos()


    def checkParticleCollision(self, other_particle):
        """Check if particle is in collision with another Particle instance.
        Checks by measuring the distance between the Particle centers, separation must be less than
        the sum of radii for a collision to occur.

        Args:
            other_particle (Particle): other particle object to reference against

        Returns:
            (bool): True if there is a collision, false if there is none
        """
        return np.linalg.norm(self.position - other_particle.position) <= (self.radius + other_particle.radius) + DIFFERENCE_EPSILON/2

    def drawSelf(self, ax):
        """Given a set of axes, draw the particle on the axes
        Color Particle green if there is an aggregate within the particle;
        Color Particle red if there is no aggregate within the particle

        Args:
            (plt.axes): Axes to draw circle onto
        """

        circle = Circle(xy=self.position, radius=self.radius, **self.drawstyle)
        ax.add_patch(circle)

        # if self.is_wrapped_around:
        #     circle = Circle(xy=self.wrap_around_position, radius=self.radius, **self.drawstyle)
        #     ax.add_patch(circle)
        return circle

    def advance(self, position_update:np.ndarray, dt:float):
        """advance particle's position given a position update vector and timestamp.

        Args:
            position_update (np.ndarray): position update
            dt (float): time interval of advancement
        """
        
        self.position += position_update
        self.velocity = position_update/dt

    def printAttr(self):
        """Prints particle attributes to console for debugging
        """
        print(f"ID: {self.ID}\tpos: {self.position}\t radius: {self.radius}")
        print(f"has_agg: {self.has_agg}")

    def isContainedInSimulation(self, lower_boundaries: Tuple[float], upper_boundaries: Tuple[float]) -> bool:
        """checks to see if particle is outside of the bounds of the simulation, to determine
        whether the particle needs to be wrapped around.

        Args:
            lower_boundaries (Tuple[float]): Contains information on lower_x, lower_y bounds
                of simulation box
            upper_boundaries (Tuple[float]): Contains information on upper_x, upper_y bounds
                of simulation box

        Returns:
            (bool): Whether if particle has crossed boundary of simulation box
        """

        # lower_x, lower_y = lower_boundaries
        # upper_x, upper_y = upper_boundaries

        lowestBound = self.position - self.radius
        highestBound = self.position + self.radius
        if all(lowestBound > lower_boundaries) and all(highestBound < upper_boundaries):
            self.is_wrapped_around = False
            self.wrapped_pos = None
            return True
        else:
            self.is_wrapped_around = True

            return False


class Box:
    def __init__(self, lower_y: float, upper_y: float, lower_x: float, upper_x: float):
        """Simulation box definition

        Args:
            lower_y (float): simulation box lower y bound
            upper_y (float): simulation box upper y bound
            lower_x (float): simulation box lower x bound
            upper_x (float): simulation box upper x bound
        """
        self.lower_y = lower_y
        self.lower_x = lower_x
        self.upper_y = upper_y
        self.upper_x = upper_x
        self.lower_bounds = np.array([self.lower_x, self.lower_y])
        self.upper_bounds = np.array([self.upper_x, self.upper_y])
        self.height = self.upper_y - self.lower_y
        self.width = self.upper_x - self.lower_x

    def isParticleInBox(self, particle_obj: Particle):
        """Check if particle is contained within the simulation boundaries supplied
        Returns:
            (bool): True if the particle is within the box, otherwise False
        """

        particle_lower_bounds = particle_obj.position - particle_obj.radius
        particle_upper_bounds = particle_obj.position + particle_obj.radius
        if all(particle_lower_bounds > self.lower_bounds) and all(particle_upper_bounds < self.upper_bounds):
            return True
        else:
            return False
