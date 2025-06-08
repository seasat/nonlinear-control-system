# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from spacecraft import Spacecraft
from simulation import Simulation
from attitude import YawPitchRoll
from orbit import Orbit
from controller import StateFeedbackController, NDIController, TSSController, INDIController


def main():
    # settings
    ORBIT = Orbit(
        gravitational_parameter = 3.98600442e14, # m^3 s^-2
        semi_major_axis = 6.378137e6 + 700e3 # m
    )
    INERTIA_TENSOR = np.asmatrix([
        [124.531, 0.0, 0.0],
        [0.0, 124.586, 0.0],
        [0.0, 0.0, 1.704]
    ]) # kg m^2
    DISTURBANCE_TORQUE = np.asmatrix([
        [1e-4],
        [1e-4],
        [1e-4],
    ]) # Nm
    INITIAL_ATTITUDE = YawPitchRoll([np.deg2rad(30), np.deg2rad(30), np.deg2rad(30)]) # rad
    SIMULATION_DURATION = 1500 # s
    SAMPLE_TIME = 0.1 # s
    ATTITUDE_COMMANDS = {
        0: YawPitchRoll([0, 0, 0]),
        100: YawPitchRoll([np.deg2rad(60), np.deg2rad(60), np.deg2rad(60)]),
        500.1: YawPitchRoll([np.deg2rad(-60), np.deg2rad(-60), np.deg2rad(-60)]),
        900.1: YawPitchRoll([np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)])
    }
    NATURAL_FREQUENCY = 0.1 # rad/s
    DAMPING_RATIO = 0.9

    #plt.rc('text', usetex=True)

    sc = Spacecraft(INERTIA_TENSOR, INITIAL_ATTITUDE, np.zeros((3, 1)), ORBIT)
    quaternion_commands = {t: att.to_quaternion() for t, att in ATTITUDE_COMMANDS.items()}

    # linear controller
    state_feedback_controller = StateFeedbackController(sc, NATURAL_FREQUENCY, DAMPING_RATIO)
    simulation = Simulation(sc, SIMULATION_DURATION, SAMPLE_TIME, DISTURBANCE_TORQUE, ATTITUDE_COMMANDS, state_feedback_controller)
    simulation.plot_attitudes()
    simulation.plot_attitude_errors()
    simulation.plot_control_torques()

    sc.set_state(INITIAL_ATTITUDE.to_quaternion(), np.zeros((3, 1)))  # Reset attitude and angular velocity for linear quaternion simulation
    state_feedback_quaternion_controller = StateFeedbackController(sc, NATURAL_FREQUENCY, DAMPING_RATIO)
    simulation_linear_quaternion = Simulation(sc, SIMULATION_DURATION, SAMPLE_TIME, DISTURBANCE_TORQUE, quaternion_commands, state_feedback_quaternion_controller)
    simulation_linear_quaternion.plot_attitudes()
    simulation_linear_quaternion.plot_attitude_errors()
    simulation_linear_quaternion.plot_control_torques()

    # ndi controller
    ndi_controller = NDIController(sc, DISTURBANCE_TORQUE, NATURAL_FREQUENCY, DAMPING_RATIO)
    sc.set_state(INITIAL_ATTITUDE, np.zeros((3, 1)))  # Reset attitude and angular velocity for NDI simulation
    simulation_ndi = Simulation(sc, SIMULATION_DURATION, SAMPLE_TIME, DISTURBANCE_TORQUE, ATTITUDE_COMMANDS, ndi_controller)
    simulation_ndi.plot_attitudes()
    simulation_ndi.plot_attitude_errors()

    sc.set_state(INITIAL_ATTITUDE.to_quaternion(), np.zeros((3, 1)))  # Reset attitude and angular velocity for NDI quaternion simulation
    ndi_quaternion_controller = NDIController(sc, DISTURBANCE_TORQUE, NATURAL_FREQUENCY, DAMPING_RATIO)
    simulation_ndi_quaternion = Simulation(sc, SIMULATION_DURATION, SAMPLE_TIME, DISTURBANCE_TORQUE, quaternion_commands, ndi_quaternion_controller)
    simulation_ndi_quaternion.plot_attitudes()
    simulation_ndi_quaternion.plot_attitude_errors()

    # tss controller
    tss_controller = TSSController(sc, DISTURBANCE_TORQUE, NATURAL_FREQUENCY, DAMPING_RATIO)
    sc.set_state(INITIAL_ATTITUDE, np.zeros((3, 1)))  # Reset attitude and angular velocity for TSS simulation
    simulation_tss = Simulation(sc, SIMULATION_DURATION, SAMPLE_TIME, DISTURBANCE_TORQUE, ATTITUDE_COMMANDS, tss_controller)
    simulation_tss.plot_attitudes()
    simulation_tss.plot_attitude_errors()

    sc.set_state(INITIAL_ATTITUDE.to_quaternion(), np.zeros((3, 1)))  # Reset attitude and angular velocity for TSS quaternion simulation
    tss_quaternion_controller = TSSController(sc, DISTURBANCE_TORQUE, NATURAL_FREQUENCY, DAMPING_RATIO)
    simulation_tss_quaternion = Simulation(sc, SIMULATION_DURATION, SAMPLE_TIME, DISTURBANCE_TORQUE, quaternion_commands, tss_quaternion_controller)
    simulation_tss_quaternion.plot_attitudes()
    simulation_tss_quaternion.plot_attitude_errors()

    # indi controller
    indi_controller = INDIController(sc, DISTURBANCE_TORQUE, NATURAL_FREQUENCY, DAMPING_RATIO)
    sc.set_state(INITIAL_ATTITUDE, np.zeros((3, 1)))  # Reset attitude and angular velocity for INDI simulation
    simulation_indi = Simulation(sc, SIMULATION_DURATION, SAMPLE_TIME, DISTURBANCE_TORQUE, ATTITUDE_COMMANDS, indi_controller)
    simulation_indi.plot_attitudes()
    simulation_indi.plot_attitude_errors()

    sc.set_state(INITIAL_ATTITUDE.to_quaternion(), np.zeros((3, 1)))  # Reset attitude and angular velocity for INDI quaternion simulation
    simulation_indi_quaternion = Simulation(sc, SIMULATION_DURATION, SAMPLE_TIME, DISTURBANCE_TORQUE, quaternion_commands, indi_controller)
    simulation_indi_quaternion.plot_attitudes()
    simulation_indi_quaternion.plot_attitude_errors()

    plt.show()


if __name__ == "__main__":
    SystemExit(main())
