#!/usr/bin/python3
from cmath import log
from concurrent.futures import thread
from enum import Enum
from math import isfinite
from ntpath import join
from socket import setdefaulttimeout
import sys
from typing import Any, Callable, Dict, List, Optional

import threading
import time
import os

import logging
from logging.handlers import RotatingFileHandler

from rtplot import client

import numpy as np
import scipy.signal
from flexsea import flexsea as flex
from flexsea import fxEnums as fxe
from flexsea import fxUtils as fxu

import signal
from math import sqrt

from pyPS4Controller.controller import Controller

# TODO: Support for TMotor driver with similar structure

PRECISION_OF_SLEEP = 0.0001

class LoopKiller:
    """
    Soft Realtime Loop---a class designed to allow clean exits from infinite loops
    with the potential for post-loop cleanup operations executing.

    The Loop Killer object watches for the key shutdown signals on the UNIX operating system (which runs on the PI)
    when it detects a shutdown signal, it sets a flag, which is used by the Soft Realtime Loop to stop iterating.
    Typically, it detects the CTRL-C from your keyboard, which sends a SIGTERM signal.

    the function_in_loop argument to the Soft Realtime Loop's blocking_loop method is the function to be run every loop.
    A typical usage would set function_in_loop to be a method of an object, so that the object could store program state.
    See the 'ifmain' for two examples.

    Author: Gray C. Thomas, Ph.D
    https://github.com/GrayThomas, https://graythomas.github.io
    """    
    def __init__(self, fade_time=0.0):
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)
        signal.signal(signal.SIGHUP, self.handle_signal)
        self._fade_time = fade_time
        self._soft_kill_time = None

    def handle_signal(self, signum, frame):
        self.kill_now = True

    def get_fade(self):
        # interpolates from 1 to zero with soft fade out
        if self._kill_soon:
            t = time.time() - self._soft_kill_time
            if t >= self._fade_time:
                return 0.0
            return 1.0 - (t / self._fade_time)
        return 1.0

    _kill_now = False
    _kill_soon = False

    @property
    def kill_now(self):
        if self._kill_now:
            return True
        if self._kill_soon:
            t = time.time() - self._soft_kill_time
            if t > self._fade_time:
                self._kill_now = True
        return self._kill_now

    @kill_now.setter
    def kill_now(self, val):
        if val:
            if self._kill_soon:  # if you kill twice, then it becomes immediate
                self._kill_now = True
            else:
                if self._fade_time > 0.0:
                    self._kill_soon = True
                    self._soft_kill_time = time.time()
                else:
                    self._kill_now = True
        else:
            self._kill_now = False
            self._kill_soon = False
            self._soft_kill_time = None

class SoftRealtimeLoop:
    """
    Soft Realtime Loop---a class designed to allow clean exits from infinite loops
    with the potential for post-loop cleanup operations executing.

    The Loop Killer object watches for the key shutdown signals on the UNIX operating system (which runs on the PI)
    when it detects a shutdown signal, it sets a flag, which is used by the Soft Realtime Loop to stop iterating.
    Typically, it detects the CTRL-C from your keyboard, which sends a SIGTERM signal.

    the function_in_loop argument to the Soft Realtime Loop's blocking_loop method is the function to be run every loop.
    A typical usage would set function_in_loop to be a method of an object, so that the object could store program state.
    See the 'ifmain' for two examples.

    Author: Gray C. Thomas, Ph.D
    https://github.com/GrayThomas, https://graythomas.github.io
    """        
    def __init__(self, dt=0.001, report=False, fade=0.0):
        self.t0 = self.t1 = time.time()
        self.killer = LoopKiller(fade_time=fade)
        self.dt = dt
        self.ttarg = None
        self.sum_err = 0.0
        self.sum_var = 0.0
        self.sleep_t_agg = 0.0
        self.n = 0
        self.report = report

    def __del__(self):
        if self.report:
            print("In %d cycles at %.2f Hz:" % (self.n, 1.0 / self.dt))
            print("\tavg error: %.3f milliseconds" % (1e3 * self.sum_err / self.n))
            print(
                "\tstddev error: %.3f milliseconds"
                % (
                    1e3
                    * sqrt((self.sum_var - self.sum_err**2 / self.n) / (self.n - 1))
                )
            )
            print(
                "\tpercent of time sleeping: %.1f %%"
                % (self.sleep_t_agg / self.time() * 100.0)
            )

    @property
    def fade(self):
        return self.killer.get_fade()

    def run(self, function_in_loop, dt=None):
        if dt is None:
            dt = self.dt
        self.t0 = self.t1 = time.time() + dt
        while not self.killer.kill_now:
            ret = function_in_loop()
            if ret == 0:
                self.stop()
            while time.time() < self.t1 and not self.killer.kill_now:
                if signal.sigtimedwait(
                    [signal.SIGTERM, signal.SIGINT, signal.SIGHUP], 0
                ):
                    self.stop()
            self.t1 += dt
        print("Soft realtime loop has ended successfully.")

    def stop(self):
        self.killer.kill_now = True

    def time(self):
        return time.time() - self.t0

    def time_since(self):
        return time.time() - self.t1

    def __iter__(self):
        self.t0 = self.t1 = time.time() + self.dt
        return self

    def __next__(self):
        if self.killer.kill_now:
            raise StopIteration

        while (
            time.time() < self.t1 - 2 * PRECISION_OF_SLEEP and not self.killer.kill_now
        ):
            t_pre_sleep = time.time()
            time.sleep(
                max(PRECISION_OF_SLEEP, self.t1 - time.time() - PRECISION_OF_SLEEP)
            )
            self.sleep_t_agg += time.time() - t_pre_sleep

        while time.time() < self.t1 and not self.killer.kill_now:
            if signal.sigtimedwait([signal.SIGTERM, signal.SIGINT, signal.SIGHUP], 0):
                self.stop()
        if self.killer.kill_now:
            raise StopIteration
        self.t1 += self.dt
        if self.ttarg is None:
            # inits ttarg on first call
            self.ttarg = time.time() + self.dt
            # then skips the first loop
            return self.t1 - self.t0
        error = time.time() - self.ttarg  # seconds
        self.sum_err += error
        self.sum_var += error**2
        self.n += 1
        self.ttarg += self.dt
        return self.t1 - self.t0

class Actpack:
    """A class that contains and works with all the sensor data from a Dephy Actpack.

    Attributes:
        motor_angle (double) : Motor angle
        motor_velocity (double : Motor velocity

    """
    MOTOR_COUNT_PER_REV = 16384
    NM_PER_AMP = 0.146
    RAD_PER_DEG = np.pi/180.0
    MOTOR_RAD_PER_COUNT = 2*np.pi/MOTOR_COUNT_PER_REV
    JOINT_RAD_PER_COUNT = 360/2**14 * RAD_PER_DEG
    MOTOR_COUNT_TO_RADIANS = lambda x: x * (np.pi/180.0/45.5111)
    RADIANS_TO_MOTOR_COUNTS = lambda q: q * (180*45.5111/np.pi)

    RAD_PER_SEC_GYROLSB = np.pi/180/32.8
    G_PER_ACCLSB = 1.0/8192

    TRANSMISSION_RATIO = 5.1 * 9.0

    CONSTANT_A = 0.00028444
    CONSTANT_C = 0.0007812

    NM_PER_RAD_TO_K = MOTOR_RAD_PER_COUNT/CONSTANT_C*1e3/NM_PER_AMP/TRANSMISSION_RATIO**2
    NM_S_PER_RAD_TO_B = RAD_PER_DEG/CONSTANT_A*1e3/NM_PER_AMP/TRANSMISSION_RATIO**2

    def __init__(self, fxs, port, baud_rate, frequency, logger: logging.Logger = None, debug_level=0) -> None:
        """
        Initializes the Actpack class

        Args:
            fxs (Fle): _description_
            port (_type_): _description_
            baud_rate (_type_): _description_
            has_loadcell (bool, optional): _description_. Defaults to False.
            debug_level (int, optional): _description_. Defaults to 0.
        """

        self._fxs = fxs
        self._dev_id = None
        self._app_type = None

        self._port = port
        self._baud_rate = baud_rate
        self._debug_level = debug_level

        self._frequency = frequency

        self._streaming = False
        self._data = None

        self.log = logger
        self._start()

    def update(self) -> None:
        """
        Updates/Reads data from the Actpack
        """
        if self.is_streaming:
            self._data = self._fxs.read_device(self._dev_id)

    def _start(self):
        try:
            self._dev_id = self._fxs.open(self._port, self._baud_rate, self._debug_level)
            self._app_type = self._fxs.get_app_type(self._dev_id)

        except OSError:
            self.log.error("Actpack is not powered ON.")
            sys.exit()

    def _start_streaming_data(self, log_en=False):
        """Starts streaming dataa

        Args:
            frequency (int, optional): _description_. Defaults to 500.
            log_en (bool, optional): _description_. Defaults to False.
        """

        self._fxs.start_streaming(self._dev_id, freq=self._frequency, log_en=log_en)
        self._streaming = True
        time.sleep(1)

    def _stop_streaming_data(self):
        """
        Stops streaming data
        """
        if self.is_streaming:
            self._fxs.stop_streaming(self._dev_id)
            self._streaming = False
            time.sleep(1)

    def _shutdown(self):
        """
        Shuts down the Actpack
        """
        self._stop_streaming_data()
        self._fxs.send_motor_command(self._dev_id, fxe.FX_NONE, 0)
        time.sleep(1/self._frequency)
        self._fxs.close(self._dev_id)
        time.sleep(1/self._frequency)

    def _set_position_gains(self, kp: int = 300, ki: int = 20, kd: int = 40, kff = 128):
        """Sets Position Gains

        Args:
            kp (int, optional): _description_. Defaults to 200.
            ki (int, optional): _description_. Defaults to 50.
            kd (int, optional): _description_. Defaults to 0.
        """
        assert(isfinite(kp) and kp >= 0 and kp <= 600)
        assert(isfinite(ki) and ki >= 0 and ki <= 100)
        assert(isfinite(kd) and kd >= 0 and kd <= 200)

        # self._set_qaxis_voltage(0)
        self._fxs.set_gains(self._dev_id, kp, ki, kd, 0, 0, kff)
        # self._set_motor_angle_counts(self.motor_angle_counts)

    def _set_current_gains(self, kp: int = 50, ki: int = 200, kff: int = 80):
        """Sets Current Gains

        Args:
            kp (int, optional): _description_. Defaults to 40.
            ki (int, optional): _description_. Defaults to 400.
            ff (int, optional): _description_. Defaults to 128.
        """
        assert(isfinite(kp) and kp >= 0 and kp <= 80)
        assert(isfinite(ki) and ki >= 0 and ki <= 800)
        assert(isfinite(kff) and kff >= 0 and kff <= 128)

        self._fxs.set_gains(self._dev_id, kp, ki, 0, 0, 0, kff)
        time.sleep(1/self._frequency)

    def _set_impedance_gains(self, K: int = 60, B: int = 100, kp: int = 50, ki: int = 200, kff: int = 80):
        """Sets Impedance Gains

        Args:
            kp (int, optional): _description_. Defaults to 40.
            ki (int, optional): _description_. Defaults to 400.
            K (int, optional): _description_. Defaults to 300.
            B (int, optional): _description_. Defaults to 1600.
            ff (int, optional): _description_. Defaults to 128.
        """
        assert(isfinite(kp) and kp >= 0 and kp <= 80)
        assert(isfinite(ki) and ki >= 0 and ki <= 800)
        assert(isfinite(kff) and kff >= 0 and kff <= 128)
        assert(isfinite(K) and K >= 0)
        assert(isfinite(B) and B >= 0)

        # self._set_qaxis_voltage(0)
        self._fxs.set_gains(self._dev_id, kp, ki, 0, K, B, kff)
        # self._set_motor_angle_radians(self.motor_angle)

    def _set_motor_angle_counts(self, position: int = 0):
        """Sets Motor Angle

        Args:
            position (int, optional): Angular position in motor counts. Defaults to 0.
        """
        assert(isfinite(position))
        self._fxs.send_motor_command(self._dev_id, fxe.FX_POSITION, position)

    def _set_motor_angle_radians(self, theta: float = 0.0):
        assert(isfinite(theta))
        self._fxs.send_motor_command(self._dev_id, fxe.FX_POSITION, int(theta/self.MOTOR_RAD_PER_COUNT))

    def _set_equilibrium_angle_counts(self, position: int = 0):
        assert(isfinite(position))
        self._fxs.send_motor_command(self._dev_id, fxe.FX_IMPEDANCE, position)

    def _set_equilibrium_angle_radians(self, theta: float = 0.0):
        assert(isfinite(theta))
        self._fxs.send_motor_command(self._dev_id, fxe.FX_IMPEDANCE, int(theta/self.MOTOR_RAD_PER_COUNT))

    def _set_voltage(self, volt: float = 0.0):
        """Sets Voltage

        Args:
            volt (int, optional): Voltage in V. Defaults to 0. Maximum limit is set to 36V.
        """
        assert(isfinite(volt) and abs(volt) <= 36)
        self._fxs.send_motor_command(self._dev_id, fxe.FX_VOLTAGE, int(volt * 1000))

    def _set_qaxis_voltage(self, volt: float = 0.0):
        """Sets Q-Axis Voltage

        Args:
            volt (int, optional): Voltage in V. Defaults to 0. Maximum limit is set to 36V.
        """
        assert(isfinite(volt) and abs(volt) <= 36)
        self._fxs.send_motor_command(self._dev_id, fxe.FX_NONE, int(volt * 1000))

    def _set_qaxis_current(self, current: float = 0.0):
        """Sets Q-Axis Current

        Args:
            current (int, optional): Current in A. Defaults to 0. Maximum limit is set to 22A.
        """
        assert(isfinite(current) and abs(current) <= 22)
        self._fxs.send_motor_command(self._dev_id, fxe.FX_CURRENT, int(current * 1000))

    def _set_motor_torque(self, torque: float = 0.0):
        self._set_qaxis_current(torque/self.NM_PER_AMP)

    @property
    def fxs(self):
        return self._fxs

    @property
    def port(self):
        return self._port

    @property
    def baud_rate(self):
        return self._baud_rate

    @property
    def is_streaming(self):
        return self._streaming

    @property
    def state_time(self):
        return self._data.state_time

    @property
    def battery_current(self):
        return self._data.batt_curr * 1e-3

    @property
    def battery_voltage(self):
        return self._data.batt_volt * 1e-3

    @property
    def motor_temperature(self):
        return self._data.temp

    @property
    def motor_current(self):
        return self._data.mot_cur * 1e-3

    @property
    def motor_angle_counts(self):
        return self._data.mot_ang

    @property
    def motor_angle(self):
        return self._data.mot_ang * self.MOTOR_RAD_PER_COUNT

    @property
    def motor_velocity(self):
        return self._data.mot_vel * self.RAD_PER_DEG

    @property
    def motor_torque(self):
        return self.motor_current * self.NM_PER_AMP

    @property
    def motor_acceleration(self):
        return self._data.mot_acc

    @property
    def joint_angle(self):
        return self._data.ank_ang * self.JOINT_RAD_PER_COUNT

    @property
    def joint_angle_counts(self):
        return self._data.ank_ang

    @property
    def joint_velocity(self):
        return self._data.ank_vel * self.RAD_PER_DEG

    @property
    def acceleration_x(self):
        return self._data.accelx * self.G_PER_ACCLSB

    @property
    def acceleration_y(self):
        return self._data.accely * self.G_PER_ACCLSB

    @property
    def acceleration_z(self):
        return self._data.accelz * self.G_PER_ACCLSB

    @property
    def gyro_x(self):
        return self._data.gyro_x * self.RAD_PER_SEC_GYROLSB

    @property
    def gyro_y(self):
        return self._data.gyro_y * self.RAD_PER_SEC_GYROLSB

    @property
    def gyro_z(self):
        return self._data.gyro_z * self.RAD_PER_SEC_GYROLSB

    @property
    def gyroscope_vector(self):
        return np.array([self._data.gyro_x, self._data.gyro_y, self._data.gyro_z]).T * self.RAD_PER_SEC_GYROLSB

    @property
    def accelerometer_vector(self):
        return np.array([self._data.accelx, self._data.accely, self._data.accelz]).T * self.G_PER_ACCLSB

    @property
    def genvars(self):
        return np.array([
            self._data.genvar_0, 
            self._data.genvar_1, 
            self._data.genvar_2, 
            self._data.genvar_3, 
            self._data.genvar_4, 
            self._data.genvar_5
            ])

class JointState(Enum):
    NEUTRAL = 0
    VOLTAGE = 1
    POSITION = 2
    CURRENT = 3
    IMPEDANCE = 4

class Joint(Actpack):

    def __init__(self, name, fxs, port, baud_rate, frequency, logger, debug_level=0) -> None:
        super().__init__(fxs, port, baud_rate, frequency, logger, debug_level)

        self._name = name
        self._filename = "./encoder_map_" + self._name + ".txt"

        self._joint_angle_array = None
        self._motor_count_array = None

        self._state = JointState.NEUTRAL

        self._k: float = 0.08922
        self._b: float = 0.0038070
        self._theta: float = 0.0

        self._kp: int = None
        self._ki: int = None
        self._kd: int = None
        self._kff: int = None

        self.initial_angle = None
        self.range_of_motion = np.deg2rad(60)
        self.trajectory = None
        self.lock_trajectory = False
        self.release_lock = False

    def home(self, save=True, homing_voltage=4.0, homing_rate=0.001):

        # TODO Logging module
        self.log.info(f"[{self._name}] Initiating Homing Routine.")

        minpos_motor, minpos_joint, _ = self._homing_routine(direction=1.0, hvolt=homing_voltage, hrate=homing_rate)
        self.log.info(
            f"[{self._name}] Minimum Motor angle: {minpos_motor}, Minimum Joint angle: {minpos_joint}"
        )
        time.sleep(0.5)
        maxpos_motor, maxpos_joint, max_output = self._homing_routine(direction=-1.0, hvolt=homing_voltage, hrate=homing_rate)
        self.log.info(
            f"[{self.name}] Maximum Motor angle: {maxpos_motor}, Maximum Joint angle: {maxpos_joint}"
        )

        max_output = np.array(max_output).reshape((len(max_output), 2))
        output_motor_count = max_output[:, 1]

        _, ids = np.unique(output_motor_count, return_index=True)

        if save:
            self._save_encoder_map(data=max_output[ids])

        self.log.info(f"[{self.name}] Homing Successfull.")

    def _homing_routine(self, direction, hvolt=4.0, hrate=0.001):
        """Homing Routine

        Args:
            direction (_type_): _description_
            hvolt (int, optional): _description_. Defaults to 2500.
            hrate (float, optional): _description_. Defaults to 0.001.

        Returns:
            _type_: _description_
        """
        output = []
        velocity_threshold = 0
        go_on = True

        self.update()
        current_motor_position = self.motor_angle_counts
        current_joint_position = self.joint_angle_counts

        self.set_state(JointState.VOLTAGE)
        self.set_voltage(direction * hvolt)
        time.sleep(0.05)
        self.update()
        cpos_motor = self.motor_angle_counts
        initial_velocity = self.joint_velocity
        output.append([self.joint_angle] + [cpos_motor])
        velocity_threshold = abs(initial_velocity / 10.0)

        while go_on:
            time.sleep(hrate)
            self.update()
            cpos_motor = self.motor_angle_counts
            cvel_joint = self.joint_velocity
            output.append([self.joint_angle] + [cpos_motor])

            if abs(cvel_joint) <= velocity_threshold:
                self.set_voltage(0)
                current_motor_position = self.motor_angle_counts
                current_joint_position = self.joint_angle_counts

                go_on = False

        return current_motor_position, current_joint_position, output

    def get_motor_angle(self, desired_joint_angle):
        """Returns Motor Count corresponding to the passed Joint angle value

        Args:
            desired_joint_angle (_type_): in Radians

        Returns:
            _type_: _description_
        """
        if self._joint_angle_array is None:
            self._load_encoder_map()

        desired_motor_count = np.interp(
            np.array(desired_joint_angle), self._joint_angle_array, self._motor_count_array
        )
        return desired_motor_count * self.MOTOR_RAD_PER_COUNT

    def set_state(self, to_state: JointState = JointState.NEUTRAL):
        self._state = to_state

    def set_current_gains(self, kp: int = 50, ki: int = 200, kff: int = 80):
        if self.state == JointState.CURRENT:
            self._kp = kp
            self._ki = ki
            self._kd = 0
            self._kff = kff

            self._set_current_gains(kp=kp, ki=ki, kff=kff)
        else:
            self.log.warning("Joint State is incorrect.")

    def set_position_gains(self, kp: int = 300, ki: int = 20, kd: int = 40, kff = 128):
        if self.state == JointState.POSITION:
            self._kp = kp
            self._ki = ki
            self._kd = kd
            self._kff = kff

            self._set_position_gains(kp=kp, ki=ki, kd=kd, kff=kff)
        else:
            self.log.warning("Joint State is incorrect.")

    def set_impedance_gains(self, k: float = 6.4454, b: float = 0.019273, kp: int = 40, ki: int = 400, kff: int = 128):

        if self.state == JointState.IMPEDANCE:

            self._k = k
            self._b = b

            self._kp = kp
            self._ki = ki
            self._kd = 0
            self._kff = kff

            self._set_impedance_gains(K=int(k*self.NM_PER_RAD_TO_K), B=int(k*self.NM_S_PER_RAD_TO_B), kp=kp, ki=ki, kff=kff)
        else:
            self.log.warning("Joint State is incorrect.")

    def update_impedance_gains(self):
        if self.state == JointState.IMPEDANCE:       
            self._set_impedance_gains(K=int(self._k*self.NM_PER_RAD_TO_K), B=int(self._b*self.NM_S_PER_RAD_TO_B), kp=self._kp, ki=self._ki, kff=self._kff)

        else:
            self.log.warning("Joint State is incorrect.")        

    def set_impedance_equilibrium(self, theta: float = None):
        if theta is None:
            self._theta = self.motor_angle
        else:
            self._theta = theta

        if self.state == JointState.IMPEDANCE:
            self._set_equilibrium_angle_radians(theta=self._theta)
        else:
            self.log.warning("Joint State is incorrect.")            

    def set_current(self, current):
        if self.state == JointState.CURRENT:
            self._set_qaxis_current(current)
        else:
            self.log.warning("Joint State is incorrect.")

    def set_position(self, position):
        if self.state == JointState.POSITION:
            self._set_motor_angle_radians(position)
        else:
            self.log.warning("Joint State is incorrect.")

    def set_voltage(self, volt):
        if self.state == JointState.VOLTAGE:
            self._set_voltage(volt)
        else:
            self.log.warning("Joint State is incorrect.")

    def increase_equilibrium_angle(self, increment=15):
        # Increment in (deg)
        if self.trajectory < self.initial_angle + self.range_of_motion:
            self.trajectory = self.equilibrium_angle + np.deg2rad(increment)
        else:
            self.trajectory = self.range_of_motion

        self.log.info(f"Increased Equilibrium Angle to {self.equilibrium_angle} rad.")   

    def decrease_equilibrium_angle(self, decrement=15):
        # Decrement in (deg)
        if self.trajectory > self.initial_angle + np.deg2rad(10):
            self.trajectory = self.equilibrium_angle - np.deg2rad(decrement)
        else:
            self.trajectory = self.initial_angle

        self.log.info(f"Decreased Equilibrium Angle to {self.equilibrium_angle} rad.")   

    def increase_stiffness(self, increment=15):
        # Increment in Nm/rad
        
        if self._k < 500/self.NM_PER_RAD_TO_K:
            self._k = self._k + increment
        else:
            self._k = 500/self.NM_PER_RAD_TO_K

        self.log.info(f"Increased Stiffness to {self.stiffness} Nm/rad.")   

    def decrease_stiffness(self, decrement=15):
        # Decrement in Nm/rad
        if self._k > 30/self.NM_PER_RAD_TO_K:
            self._k = self._k - decrement
        else:
            self._k = 15/self.NM_PER_RAD_TO_K

        self.log.info(f"Decreased Stiffness to {self.stiffness} Nm/rad.")

    def increase_damping(self, increment=0.085):
        if self._b < 300/self.NM_S_PER_RAD_TO_B:
            self._b = self._b + increment
        else:
            self._b = 300/self.NM_S_PER_RAD_TO_B

        self.log.info(f"Increased Damping to {self.damping} Nm/s/rad.")

    def decrease_damping(self, decrement=0.085):
        if self._b > 10/self.NM_S_PER_RAD_TO_B:
            self._b = self._b - decrement
        else:
            self._b = 10/self.NM_S_PER_RAD_TO_B

        self.log.info(f"Decreased Damping to {self.damping} Nm/s/rad.")

    def _save_encoder_map(self, data):
        """
        Saves encoder_map: [Joint angle, Motor count] to a text file
        """
        np.savetxt(self._filename, data, fmt="%.5f")

    def _load_encoder_map(self):
        """
        Loads Joint angle array, Motor count array, Min Joint angle, and Max Joint angle
        """
        data = np.loadtxt(self._filename, dtype=np.float64)
        self._joint_angle_array = data[:, 0]
        self._motor_count_array = np.array(data[:, 1], dtype=np.int32)

        self._min_joint_angle = np.min(self._joint_angle_array)
        self._max_joint_angle = np.max(self._joint_angle_array)

        self._joint_angle_array = self._max_joint_angle - self._joint_angle_array

        # Applying a median filter with a kernel size of 3
        self._joint_angle_array = scipy.signal.medfilt(
            self._joint_angle_array, kernel_size=3
        )
        self._motor_count_array = scipy.signal.medfilt(
            self._motor_count_array, kernel_size=3
        )

    @property
    def name(self):
        return self._name

    @property
    def state(self):
        return self._state

    @property
    def stiffness(self):
        return self._k

    @property
    def damping(self):
        return self._b

    @property
    def equilibrium_angle(self):
        return self._theta

    @property
    def proportional_gain(self):
        return self._kp

    @property
    def integral_gain(self):
        return self._ki

    @property
    def derivative_gain(self):
        return self._kd    

class Loadcell:
    def __init__(self, joint: Joint, amp_gain: float = 125.0, exc: float = 5.0, loadcell_matrix = None, logger: logging.Logger = None) -> None:
        self._joint = joint
        self._amp_gain = 125.0
        self._exc = 5.0

        if not loadcell_matrix:
            self._loadcell_matrix = np.array(
                [
                    (-38.72600, -1817.74700, 9.84900, 43.37400, -44.54000, 1824.67000),
                    (-8.61600, 1041.14900, 18.86100, -2098.82200, 31.79400, 1058.6230),
                    (
                        -1047.16800,
                        8.63900,
                        -1047.28200,
                        -20.70000,
                        -1073.08800,
                        -8.92300,
                    ),
                    (20.57600, -0.04000, -0.24600, 0.55400, -21.40800, -0.47600),
                    (-12.13400, -1.10800, 24.36100, 0.02300, -12.14100, 0.79200),
                    (-0.65100, -28.28700, 0.02200, -25.23000, 0.47300, -27.3070),
                ]
            )
        else:
            self._loadcell_matrix = loadcell_matrix

        self._loadcell_data = None
        self._loadcell_zero = np.zeros((1, 6), dtype=np.double)
        self._zeroed = False
        self.log = logger

    def reset(self):
        self._zeroed = False
        self._loadcell_zero = np.zeros((1, 6), dtype=np.double)

    def update(self, loadcell_zero = None):
        """
        Computes Loadcell data

        """
        loadcell_signed = (self._joint.genvars - 2048) / 4095 * self._exc
        loadcell_coupled = loadcell_signed * 1000 / (self._exc * self._amp_gain)

        if loadcell_zero is None:
            self._loadcell_data = np.transpose(self._loadcell_matrix.dot(np.transpose(loadcell_coupled))) - self._loadcell_zero
        else:
            self._loadcell_data = np.transpose(self._loadcell_matrix.dot(np.transpose(loadcell_coupled))) - loadcell_zero


    def initialize(self, number_of_iterations: int = 2000):
        """
        Obtains the initial loadcell reading (aka) loadcell_zero
        """
        ideal_loadcell_zero = np.zeros((1, 6), dtype=np.double)
        if not self._zeroed:
            if self._joint.is_streaming:
                self._joint.update()
                self.update()
                self._loadcell_zero = self._loadcell_data

                for _ in range(number_of_iterations):
                    self.update(ideal_loadcell_zero)
                    loadcell_offset = self._loadcell_data
                    self._loadcell_zero = (loadcell_offset + self._loadcell_zero) / 2.0

        elif input('Do you want to re-initialize loadcell? (Y/N)') == 'Y':
                self.reset()
                self.initialize()

    @property
    def is_zeroed(self):
        return self._zeroed

    @property
    def fx(self):
        return self._loadcell_data[0][0]

    @property
    def fy(self):
        return self._loadcell_data[0][1]

    @property
    def fz(self):
        return self._loadcell_data[0][2]

    @property
    def mx(self):
        return self._loadcell_data[0][3]

    @property
    def my(self):
        return self._loadcell_data[0][4]

    @property
    def mz(self):
        return self._loadcell_data[0][5]

class OSLV2:
    """
    The OSL class
    """

    def __init__(self, frequency: int = 200, log_data = False) -> None:

        self._fxs = flex.FlexSEA()
        self._loadcell = None

        self.joints: List[Joint] = []

        self._knee_id = None
        self._ankle_id = None

        self._frequency = frequency
        self.loop = SoftRealtimeLoop(dt=1/self._frequency, report=True, fade=0.1)

        #--------------------------------------------

        self._log_data = log_data
        self._log_filename = "oslrun.csv"
        self.log = logging.getLogger(__name__)

        if log_data:
            self.log.setLevel(logging.DEBUG)
        else:
            self.log.setLevel(logging.INFO)

        self._std_formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')

        self._file_handler = RotatingFileHandler(self._log_filename, mode='w', maxBytes=2000000, backupCount=3)
        self._file_handler.setLevel(logging.DEBUG)
        self._file_handler.setFormatter(self._std_formatter)

        self._stream_handler = logging.StreamHandler()
        self._stream_handler.setLevel(logging.INFO)
        self._stream_handler.setFormatter(self._std_formatter)

        self.log.addHandler(self._stream_handler)
        self.log.addHandler(self._file_handler)

        #----------------------------------------------

    def __enter__(self):
        for joint in self.joints:
            joint._start_streaming_data()

        if self._loadcell is not None:
            self.loadcell.initialize()

    def __exit__(self, type, value, tb):
        osl.log.info("Exiting control loop.")
        for joint in self.joints:
            joint.set_state()
            joint._shutdown()

    def add_joint(self, name: str, port, baud_rate, debug_level=0):

        if 'knee' in name.lower():
            self._knee_id = len(self.joints)
        elif 'ankle' in name.lower():
            self._ankle_id = len(self.joints)
        else:
            sys.exit("Joint can't be identified, kindly check the given name.")

        self.joints.append(Joint(name=name, fxs=self._fxs, port=port, baud_rate=baud_rate, frequency=self._frequency, logger=self.log, debug_level=debug_level))


    def add_loadcell(self, joint: Joint, amp_gain: float = 125.0, exc: float = 5.0, loadcell_matrix = None):
        self._loadcell = Loadcell(joint=joint, amp_gain=amp_gain, exc=exc, loadcell_matrix=loadcell_matrix, logger=self.log)

    def update(self):
        for joint in self.joints:
            joint.update()

        if self._loadcell is not None:
            self.loadcell.update()

    def home(self):
        for joint in self.joints:
            joint.home()     

    def clear_terminal(self):
        fxu.clear_terminal()

    def kill_now(self):
        self.loop.killer._kill_now = True

    @property
    def loadcell(self):
        if self._loadcell is not None:
            return self._loadcell
        else:
            sys.exit("Loadcell not connected.")

    @property
    def knee(self):
        if self._knee_id is not None:
            return self.joints[self._knee_id]
        else:
            sys.exit("Knee is not connected.")

    @property
    def ankle(self):
        if self._ankle_id is not None:
            return self.joints[self._ankle_id]
        else:
            sys.exit("Ankle is not connected.")

    @property
    def kill(self):
        return self.loop.killer._kill_soon

class OSLController(Controller):

    TRIGGERMAXABS = 32767

    def __init__(self, osl: OSLV2=None, interface="/dev/input/js0", connecting_using_ds4drv=False, event_definition=None, event_format=None):
        super().__init__(interface, connecting_using_ds4drv, event_definition, event_format)
        self.osl = osl

    def on_options_press(self):
        self.osl.kill_now()
        sys.exit()

    def on_up_arrow_press(self):
        pass

    def on_right_arrow_press(self):
        pass

    def on_down_arrow_press(self):
        pass

    def on_left_arrow_press(self):
        pass

    def on_R2_press(self, value):    
        pass    
        # if self.osl.knee.lock_trajectory:
        #     self.osl.knee.trajectory = self.osl.knee.trajectory
        # else:
        #     raw = (value + self.TRIGGERMAXABS)/(2 * self.TRIGGERMAXABS)
        #     self.osl.knee.trajectory = self.osl.knee.initial_angle + self.osl.knee.range_of_motion * self.osl.knee.TRANSMISSION_RATIO * raw

    def on_L2_press(self, value):
        pass
        # raw = (value + self.TRIGGERMAXABS)/(2 * self.TRIGGERMAXABS)
        # delta = self.osl.knee.trajectory - self.osl.knee.initial_angle

        # if not self.osl.knee.lock_trajectory and self.osl.knee.release_lock:
        #     delta = self.osl.knee.trajectory - self.osl.knee.initial_angle
        #     self.osl.knee.trajectory = self.osl.knee.trajectory - delta * (1 - raw)

        # if not self.osl.knee.lock_trajectory and raw == 1.0:
        #     self.osl.knee.lock_trajectory = True
        #     self.osl.knee.release_lock = False

        # elif self.osl.knee.lock_trajectory and raw == 1.0:
        #     self.osl.knee.lock_trajectory = False
        #     self.osl.knee.release_lock = True

        # elif self.osl.knee.release_lock and raw < 0.1:
        #     self.osl.knee.release_lock = False

    def on_R2_release(self):
        pass

    def on_L2_release(self):
        pass

    def on_R1_press(self):
        self.osl.knee.increase_stiffness()     

    def on_L1_press(self):
        self.osl.knee.decrease_stiffness()

    def on_R1_release(self):
        pass

    def on_L1_release(self):
        pass

    def on_circle_press(self):
        self.osl.knee.increase_damping()

    def on_square_press(self):
        self.osl.knee.decrease_damping()

    def on_circle_release(self):
        pass

    def on_square_release(self):
        pass

    def on_triangle_press(self):
        # self.osl.knee.increase_equilibrium_angle()
        pass

    def on_x_press(self):
        # self.osl.knee.decrease_equilibrium_angle()
        pass

    def on_triangle_release(self):
        pass

    def on_x_release(self):
        pass

    def on_R3_press(self):
        pass

    def on_L3_press(self):
        pass

    def on_R3_release(self):
        pass

    def on_L3_release(self):
        pass


if __name__ == "__main__":
    start = time.perf_counter()

    freq = 200
    MAX_CURRENT = 20

    osl = OSLV2(frequency=freq, log_data=True)
    controller = OSLController(osl=osl, interface="/dev/input/js0", connecting_using_ds4drv=False)

    osl.add_joint(name='Knee', port='/dev/ttyACM0', baud_rate=230400)
    # osl.add_loadcell(osl.knee, amp_gain=125, exc=5)

    position_plot = {'names': ['Trajectory', 'Value'],
                    'title': "Angular Position",
                    'ylabel': "Rad",
                    'xlabel': "Time",
                    'colors' : ["r","b"],
                    'line_width': [2]*2,
                    }

    stiffness_plot = {'names': ['Stiffness'],
                    'title': "Joint Stiffness",
                    'ylabel': "Nm/rad",
                    'xlabel': "Time",
                    'colors' : ["g"],
                    'line_width': [2]*2,
                    }

    damping_plot = {'names': ['Damping'],
                    'title': "Joint Damping",
                    'ylabel': "Nm/s/rad",
                    'xlabel': "Time",
                    'colors' : ["y"],
                    'line_width': [2]*2,
                    }

    plot_config = [position_plot, stiffness_plot, damping_plot]
    client.initialize_plots(plot_config)

    controller_thread = threading.Thread(target=controller.listen, args=(10,))
    controller_thread.start()

    with osl:
        osl.update()
        osl.knee.initial_angle = osl.knee.motor_angle
        osl.knee.trajectory = osl.knee.initial_angle

        osl.knee.set_state(JointState.IMPEDANCE)
        time.sleep(1/freq)

        osl.knee.set_impedance_gains()

        for t in osl.loop:
            if osl.kill:
                break

            osl.update()
            # osl.log.info(f"{osl.knee.initial_angle}, {osl.knee.trajectory}")
            osl.knee.set_impedance_equilibrium(osl.knee.trajectory)
            osl.knee.update_impedance_gains()

            if osl.knee.motor_current > MAX_CURRENT:
                osl.log.warning("High current detected, shutting down.")
                break

            data = [osl.knee.trajectory, osl.knee.motor_angle, osl.knee.stiffness, osl.knee.damping]
            client.send_array(data)

    finish = time.perf_counter()
