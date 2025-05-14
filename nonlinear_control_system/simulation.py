# -*- coding: utf-8 -*-

from spacecraft import Spacecraft


class Simulation:
    def __init__(self, spacecraft: Spacecraft, duration: float, sample_time: float) -> None:
        """
        Initialize the Simulation class with a spacecraft, duration, and sample time.

        :param spacecraft: The spacecraft to be simulated.
        :param duration: The total duration of the simulation in seconds.
        :param sample_time: The time interval between each simulation step in seconds.
        """
        assert isinstance(spacecraft, Spacecraft), "Spacecraft must be an instance of Spacecraft"

        self.spacecraft = spacecraft
        self.duration = duration
        self.sample_time = sample_time
