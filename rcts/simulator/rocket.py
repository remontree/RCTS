from canard import Canard
from tail import Tail
from engine import Engine
from segment import Segment
from environment import Environment
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
import math
import pandas as pd
import copy
pd.options.display.float_format = '{:.5f}'.format

class Rocket:
    def __init__(self, cg, cp, mass, I, pitch_init_angle, yaw_init_angle, air_density, wind_velocity, dt, canard_x, canard_area, Cl_canard_path, Cd_canard_path,Canard_Max_Angle,  tail_x, tail_area, Cl_tail_path, Cd_tail_path,
                 x, fuel_M, fuel_M_F, fuel_length, fuel_r1, fuel_r2, thrust_data_path, pressure_data_path, kp, ki, kd, R, Launcher_length):
        self.cg = cg # 연료를 넣기 전 로켓의 질량 중심
        self.cp = cp # 로켓의 압력 중심
        self.mass = mass # 로켓의 질량 (연료 제외)
        self.rocket_state_pitch = [pitch_init_angle, 0] # 현재 각도, 각속도
        self.rocket_state_yaw = [yaw_init_angle, 0] # 현재 각도, 각속도
        self.coordinate = [0,0,0] # 현재 로켓의 좌표 pitch-axis, yaw-axis, altitude-axis

        self.environment = Environment(air_density, wind_velocity, dt)

        self.pitch_canard = Canard(canard_x, canard_area, Cl_canard_path, Cd_canard_path, Canard_Max_Angle, "pitch")
        self.yaw_canard = Canard(canard_x, canard_area, Cl_canard_path, Cd_canard_path, Canard_Max_Angle, "yaw")

        self.pitch_tail = Tail(tail_x, tail_area, Cl_tail_path, Cd_tail_path, "pitch")
        self.yaw_tail = Tail(tail_x, tail_area, Cl_tail_path, Cd_tail_path, "yaw")

        self.engine = Engine(x, fuel_M, fuel_M_F, fuel_length, fuel_r1, fuel_r2, dt, thrust_data_path, pressure_data_path)

        self.cg_real = (self.cg*self.mass + self.engine.d*self.engine.current_fuel_mass)/(self.mass+self.engine.current_fuel_mass)

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.I_rocket = I
        self.I_total = self.engine.fuel_I + self.engine.current_fuel_mass*((self.engine.d-self.cg_real)**2) + self.I_rocket + self.mass*((self.cg_real-self.cg)**2)

        # pid 제어를 위한 파라메터
        self.desired_angle = 0

        self.prev_error_pitch = None
        self.integral_error_pitch = 0

        self.prev_error_yaw = None
        self.integral_error_yaw = 0

        self.Launcher_length = Launcher_length


        self.rocket_vel = 0

        self.R = R

        self.pitch_canard_lift = 0
        self.yaw_canard_lift = 0

        self.pitch_tail_lift = 0
        self.yaw_tail_lift = 0

        self.pitch_canard_drag = 0
        self.yaw_canard_drag = 0

        self.pitch_tail_drag = 0
        self.yaw_tail_drag = 0

        self.total_force = 0

        self.ascending_time = 0


        self.simulation_flag = True

        self.R1 = np.identity(3)
        self.R2 = np.identity(3)

        self.rocket_head_axis = np.array([0, 0, 1])
        self.pitch_axis = np.array([1, 0, 0])
        self.yaw_axis = np.array([0, 1, 0])

        self.rocket_vel_array = np.array([0, 0, 0], dtype=np.float64)

        self.pos = []
        self.dir = []
        self.force_dir = []

        self.x_log = []
        self.y_log = []
        self.z_log = []

        self.pitch_angle_log = []
        self.yaw_angle_log = []

        self.velocity_x = []
        self.velocity_y = []
        self.velocity_z = []

        self.thrust_log = []

        self.pitch_pid_log = []
        self.yaw_pid_log = []

        self.pitch_canard_lift_log = []
        self.pitch_tail_lift_log = []

        self.pitch_torque_log = []
        self.yaw_torque_log = []
        self.total_torque_log = []
        self.accelaration_log = []

        self.pitch_angluar_velocity_log = []
        self.yaw_angluar_velocity_log = []


        self.Is_descending = False
        self.Is_ascending = False

        # 로켓 로그 to csv
        self.rocket_angle_log = []
        self.windvector_log = []
        self.canard_angle_log = []
        self.rocekt_velocity_log = []
        self.rocekt_angular_velocity_log = []
        self.canard_relative_wind_log = []
        self.tail_relative_wind_log = []
        self.aoa_log = []
        self.lift_log = []
        self.tail_lift_log = []
        self.drag_log = []
        self.tail_drag_log = []
        self.final_wind_log = []
        self.force_log = []
        self.thrust_csv_log = []
        self.accelaration_csv_log = []
        self.cg_log = []
        self.torque_log = []

        # 비교군 로그
        self.cmp_time = []
        self.cmp_alt = []
        self.cmp_vel = []
        self.cmp_acc = []


        # DQN
        self.abs_integral_pitch = 0
        self.abs_integral_yaw = 0
    
    def parameterUpdate(self, cg, cp, mass, I, pitch_init_angle, yaw_init_angle, air_density, wind_velocity, dt, canard_x, canard_area, Cl_canard_path, Cd_canard_path,Canard_Max_Angle,  tail_x, tail_area, Cl_tail_path, Cd_tail_path,
                 x, fuel_M, fuel_M_F, fuel_length, fuel_r1, fuel_r2, thrust_data_path, pressure_data_path, kp, ki, kd, R, Launcher_length):
        self.cg = cg # 연료를 넣기 전 로켓의 질량 중심
        self.cp = cp # 로켓의 압력 중심
        self.mass = mass # 로켓의 질량 (연료 제외)
        self.rocket_state_pitch = [pitch_init_angle, 0] # 현재 각도, 각속도
        self.rocket_state_yaw = [yaw_init_angle, 0] # 현재 각도, 각속도
        self.coordinate = [0,0,0] # 현재 로켓의 좌표 pitch-axis, yaw-axis, altitude-axis

        self.environment = Environment(air_density, wind_velocity, dt)

        self.pitch_canard = Canard(canard_x, canard_area, Cl_canard_path, Cd_canard_path, Canard_Max_Angle, "pitch")
        self.yaw_canard = Canard(canard_x, canard_area, Cl_canard_path, Cd_canard_path, Canard_Max_Angle, "yaw")

        self.pitch_tail = Tail(tail_x, tail_area, Cl_tail_path, Cd_tail_path, "pitch")
        self.yaw_tail = Tail(tail_x, tail_area, Cl_tail_path, Cd_tail_path, "yaw")

        #self.engine = Engine(x, fuel_M, fuel_M_F, fuel_length, fuel_r1, fuel_r2, dt, thrust_data_path, pressure_data_path)
        self.engine.updateparam(x, fuel_M, fuel_M_F, fuel_length, fuel_r1, fuel_r2, dt, thrust_data_path, pressure_data_path)

        self.cg_real = (self.cg*self.mass + self.engine.d*self.engine.current_fuel_mass)/(self.mass+self.engine.current_fuel_mass)

        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.I_rocket = I
        self.I_total = self.engine.fuel_I + self.engine.current_fuel_mass*((self.engine.d-self.cg_real)**2) + self.I_rocket + self.mass*((self.cg_real-self.cg)**2)

        # pid 제어를 위한 파라메터
        self.desired_angle = 0

        self.prev_error_pitch = None
        self.integral_error_pitch = 0

        self.prev_error_yaw = None
        self.integral_error_yaw = 0

        self.Launcher_length = Launcher_length


        self.rocket_vel = 0

        self.R = R

        self.pitch_canard_lift = 0
        self.yaw_canard_lift = 0

        self.pitch_tail_lift = 0
        self.yaw_tail_lift = 0

        self.pitch_canard_drag = 0
        self.yaw_canard_drag = 0

        self.pitch_tail_drag = 0
        self.yaw_tail_drag = 0

        self.total_force = 0

        self.ascending_time = 0


        self.simulation_flag = True

        self.R1 = np.identity(3)
        self.R2 = np.identity(3)

        self.rocket_head_axis = np.array([0, 0, 1])
        self.pitch_axis = np.array([1, 0, 0])
        self.yaw_axis = np.array([0, 1, 0])

        self.rocket_vel_array = np.array([0, 0, 0], dtype=np.float64)

        self.pos = []
        self.dir = []
        self.force_dir = []

        self.x_log = []
        self.y_log = []
        self.z_log = []

        self.pitch_angle_log = []
        self.yaw_angle_log = []

        self.velocity_x = []
        self.velocity_y = []
        self.velocity_z = []

        self.thrust_log = []

        self.pitch_pid_log = []
        self.yaw_pid_log = []

        self.pitch_canard_lift_log = []
        self.pitch_tail_lift_log = []

        self.pitch_torque_log = []
        self.yaw_torque_log = []
        self.total_torque_log = []
        self.accelaration_log = []

        self.pitch_angluar_velocity_log = []
        self.yaw_angluar_velocity_log = []


        self.Is_descending = False
        self.Is_ascending = False




        # 로켓 로그 to csv
        self.rocket_angle_log = []
        self.windvector_log = []
        self.canard_angle_log = []
        self.rocekt_velocity_log = []
        self.rocekt_angular_velocity_log = []
        self.canard_relative_wind_log = []
        self.tail_relative_wind_log = []
        self.aoa_log = []
        self.lift_log = []
        self.tail_lift_log = []
        self.drag_log = []
        self.tail_drag_log = []
        self.final_wind_log = []
        self.force_log = []
        self.thrust_csv_log = []
        self.accelaration_csv_log = []
        self.cg_log = []
        self.torque_log = []

        
        # DQN
        self.abs_integral_pitch = 0
        self.abs_integral_yaw = 0
        
    # pid 제어
    def PID(self, theta_pitch, theta_yaw):
        theta_pitch*=(180/math.pi)
        theta_yaw*=(180/math.pi)


        error_pitch = self.desired_angle - theta_pitch

        # DQN
        self.abs_integral_pitch += abs(error_pitch * self.environment.dt * 30)

        self.integral_error_pitch += (error_pitch * self.environment.dt * 30)

        if self.prev_error_pitch == None:
            derivative_error_pitch = 0
        else:
            derivative_error_pitch = (error_pitch - self.prev_error_pitch) / (self.environment.dt*30)
        self.prev_error_pitch = error_pitch
        delta_pitch = self.kp * error_pitch + self.ki * self.integral_error_pitch + self.kd * derivative_error_pitch

        error_yaw = self.desired_angle - theta_yaw

        #DQN
        self.abs_integral_yaw += abs(error_yaw * self.environment.dt * 30)

        self.integral_error_yaw += (error_yaw * (self.environment.dt*30))
        if self.prev_error_yaw == None:
            derivative_error_yaw = 0
        else:
            derivative_error_yaw = (error_yaw - self.prev_error_yaw) / (self.environment.dt*30)
        self.prev_error_yaw = error_yaw

        delta_yaw = self.kp * error_yaw + self.ki * self.integral_error_yaw + self.kd * derivative_error_yaw


        delta_pitch = int(delta_pitch)
        delta_yaw = int(delta_yaw)

        delta_pitch*=(math.pi/180)
        delta_yaw*=(math.pi/180)
    
        return delta_pitch, delta_yaw
    

    def Force(self):

        #pitch x축에 대한 회전 행렬 R1
        self.R1 = np.array([
            [1, 0, 0],
            [0, math.cos(self.rocket_state_pitch[0]), math.sin(self.rocket_state_pitch[0])],
            [0, -math.sin(self.rocket_state_pitch[0]), math.cos(self.rocket_state_pitch[0])]
        ])

        #yaw y축에 대한 회전 행렬 R2
        self.R2 = np.array([
            [math.cos(self.rocket_state_yaw[0]), 0, -math.sin(self.rocket_state_yaw[0])],
            [0, 1, 0],
            [math.sin(self.rocket_state_yaw[0]), 0, math.cos(self.rocket_state_yaw[0])]
        ])

        #회전된 좌표축 벡터 계산
        self.rocket_head_axis = np.dot(np.dot(self.R1, self.R2), np.array([0, 0, 1]))
        self.pitch_axis = np.dot(np.dot(self.R1, self.R2), np.array([1, 0, 0]))
        self.yaw_axis = np.dot(np.dot(self.R1, self.R2), np.array([0, 1, 0]))

        #각 힘 요소 계산
        thrust = self.engine.thrust

        # 로켓 몸체가 받는 항력 계산
        drag_force = 0.5 * 0.75 * self.environment.air_density * np.pi * self.R**2 * self.rocket_vel**2

        # 카나드 날개가 받는 AOA 계산
        axis_vector = [self.rocket_head_axis, self.pitch_axis, self.yaw_axis]

        self.pitch_canard.compute_Alpha(axis_vector, self.rocket_state_pitch, self.rocket_state_yaw, self.rocket_vel_array, self.environment, self.cg_real-self.pitch_canard.d)
        self.yaw_canard.compute_Alpha(axis_vector, self.rocket_state_pitch, self.rocket_state_yaw, self.rocket_vel_array, self.environment, self.cg_real-self.yaw_canard.d)

        # 꼬리 날개가 받는 AOA 계산
        self.pitch_tail.compute_Alpha(axis_vector, self.rocket_state_pitch, self.rocket_state_yaw, self.rocket_vel_array, self.environment, self.cp-self.cg_real)
        self.yaw_tail.compute_Alpha(axis_vector, self.rocket_state_pitch, self.rocket_state_yaw, self.rocket_vel_array, self.environment, self.cp-self.cg_real)


        # 카나드, 꼬리 날개 양력 계산
        self.pitch_canard_lift = self.pitch_canard.compute_lift_force(self.environment) * 2
        self.yaw_canard_lift = self.yaw_canard.compute_lift_force(self.environment) * 2
        self.pitch_tail_lift = self.pitch_tail.compute_lift_force(self.environment) * 2
        self.yaw_tail_lift = self.yaw_tail.compute_lift_force(self.environment) * 2

        # 로켓 로그 to csv
        self.canard_relative_wind_log.append(self.pitch_canard.relative_velocity)
        self.tail_relative_wind_log.append(self.pitch_tail.relative_velocity)
        self.aoa_log.append(self.pitch_canard.Alpha*180/np.pi)
        self.lift_log.append(self.pitch_canard_lift)
        self.tail_lift_log.append(self.pitch_tail_lift)
        self.drag_log.append(self.pitch_canard_drag)
        self.tail_drag_log.append(self.pitch_tail_drag)
        self.final_wind_log.append(self.pitch_canard.final_wind_log)
        self.thrust_csv_log.append(thrust)


        # 카나드, 꼬리 날개 항력 계산
        self.pitch_canard_drag = self.pitch_canard.compute_drag_force(self.environment) * 2
        self.yaw_canard_drag = self.yaw_canard.compute_drag_force(self.environment) * 2

        self.pitch_tail_drag = self.pitch_tail.compute_drag_force(self.environment) * 2
        self.yaw_tail_drag = self.yaw_tail.compute_drag_force(self.environment) * 2

        self.pitch_canard_lift_log.append(self.pitch_canard_lift)
        self.pitch_tail_lift_log.append(self.pitch_tail_lift)
        # 중력 벡터 계산
        gravity_force = np.array([0, 0, -self.environment.g * (self.mass + self.engine.current_fuel_mass)])


        if thrust < -gravity_force[2] and self.Is_ascending==False:
            self.total_force = [0,0,0]
            self.thrust_log.append(0)

        elif self.coordinate[2] <= self.Launcher_length:
            self.total_force = (
            (thrust - drag_force - self.pitch_canard_drag  - self.yaw_canard_drag - self.pitch_tail_drag - self.yaw_tail_drag )*self.rocket_head_axis  + gravity_force
            )

            self.thrust_log.append(self.total_force[2])
        else:
            #힘의 총합 행렬 계산
            """
                (thrust  - drag_force - self.pitch_canard_drag - self.yaw_canard_drag - self.pitch_tail_drag  - self.yaw_tail_drag) * self.rocket_head_axis
                + gravity_force + self.pitch_canard_lift * self.yaw_axis + self.yaw_canard_lift * (-1) * self.pitch_axis  +
                self.pitch_tail_lift * self.yaw_axis + self.yaw_tail_lift * (-1) * self.pitch_axis
                """
            self.total_force = (
                (thrust  - drag_force - self.pitch_canard_drag - self.yaw_canard_drag - self.pitch_tail_drag  - self.yaw_tail_drag) * self.rocket_head_axis
                + gravity_force + self.pitch_canard_lift * self.yaw_axis + self.yaw_canard_lift * (-1) * self.pitch_axis  +
                self.pitch_tail_lift * self.yaw_axis + self.yaw_tail_lift * (-1) * self.pitch_axis
            )

            self.thrust_log.append(self.total_force[2])
        
        self.force_dir.append(self.total_force)

    # 병진 운동 업데이트
    def Translational(self):
        #로켓 속도 갱신
        acceleration = self.total_force / (self.mass + self.engine.current_fuel_mass)
        self.rocket_vel_array += acceleration * self.environment.dt
        #self.rocket_vel = np.linalg.norm(self.rocket_vel_array)

        self.accelaration_log.append(acceleration[2])
        if acceleration[2]>=10:
            self.Is_ascending = True
            self.ascending_time = self.environment.current_time

        self.velocity_x.append(self.rocket_vel_array[0])
        self.velocity_y.append(self.rocket_vel_array[1])
        self.velocity_z.append(self.rocket_vel_array[2])

        

        # 로켓 로그 to cvs
        self.rocekt_velocity_log.append((self.rocket_vel_array[0],self.rocket_vel_array[1], self.rocket_vel_array[2]))
        self.force_log.append(self.total_force)
        self.accelaration_csv_log.append(acceleration)

        #위치 좌표 갱신
        self.coordinate[0] += self.rocket_vel_array[0] * self.environment.dt
        self.coordinate[1] += self.rocket_vel_array[1] * self.environment.dt
        self.coordinate[2] += self.rocket_vel_array[2] * self.environment.dt
        
        # 비교군 로그 to csv
        self.cmp_vel.append(self.rocket_vel_array[2])
        self.cmp_acc.append(acceleration[2])
        self.cmp_alt.append(self.coordinate[2])
        

        self.pos.append(np.array(self.coordinate))
        self.dir.append(self.rocket_head_axis)

        self.x_log.append(self.coordinate[0])
        self.y_log.append(self.coordinate[1])
        self.z_log.append(self.coordinate[2])

    # 회전 운동 업데이트
    def Rotational(self):
        #if self.coordinate[2] <= self.Launcher_length:
        if self.coordinate[2] <= self.Launcher_length:
            pitch_total_torque = 0
            yaw_total_torque = 0
        else:
            pitch_axis = np.array([1,0,0])
            yaw_axis = np.array([0,1,0])
            canard_lift = self.pitch_canard_lift * self.yaw_axis + self.yaw_canard_lift * (-1) * self.pitch_axis
            tail_lift = self.pitch_tail_lift * self.yaw_axis + self.yaw_tail_lift * (-1) * self.pitch_axis
            # pitch 최종 토크
            pitch_total_torque = (self.cg_real - self.pitch_canard.d) * self.pitch_canard_lift + (self.cg_real - self.cp) * self.pitch_tail_lift
            # yaw 최종 토크
            yaw_total_torque = (self.cg_real - self.yaw_canard.d) * self.yaw_canard_lift + (self.cg_real - self.cp) * self.yaw_tail_lift

        self.pitch_torque_log.append(self.pitch_canard_lift)
        self.yaw_torque_log.append(self.pitch_tail_lift)
        self.total_torque_log.append(pitch_total_torque)

        pitch_angle_acc = pitch_total_torque / self.I_total
        yaw_angle_acc = yaw_total_torque / self.I_total
        self.rocket_state_pitch[1] += pitch_angle_acc * self.environment.dt
        self.rocket_state_pitch[0] += self.rocket_state_pitch[1] * self.environment.dt

        self.rocket_state_yaw[1] += yaw_angle_acc * self.environment.dt
        self.rocket_state_yaw[0] += self.rocket_state_yaw[1] * self.environment.dt


        self.pitch_angluar_velocity_log.append(self.rocket_state_pitch[1])
        self.yaw_angluar_velocity_log.append(self.rocket_state_yaw[1])
        # 로켓 to csv log
        self.cg_log.append(self.cg_real)
        self.torque_log.append(pitch_total_torque)
        self.rocket_angle_log.append(self.rocket_state_pitch[0]*(180/np.pi))
        self.windvector_log.append(self.environment.wind)
        self.rocekt_angular_velocity_log.append(self.rocket_state_pitch[1])

        self.pitch_angle_log.append(self.rocket_state_pitch[0]*(180/np.pi))
        self.yaw_angle_log.append(self.rocket_state_yaw[0]*(180/np.pi))

        if abs(self.rocket_state_pitch[0]*(180/np.pi)) >= 75 or abs(self.rocket_state_yaw[0]*(180/np.pi))>=75:
            self.Is_descending = True

    def updateEngineState(self):
        self.engine.updateEngineState(self.environment)
        self.cg_real = (self.cg*self.mass + self.engine.d*self.engine.current_fuel_mass)/(self.mass+self.engine.current_fuel_mass)
        #self.I_total = self.engine.fuel_I + self.engine.current_fuel_mass*((self.engine.d-self.cg_real)**2) + self.I_rocket + self.mass*((self.cg_real-self.cg)**2)
        self.I_total = self.I_rocket

    def updateCanard(self):
        if self.environment.dtcnt == 30 and self.Is_ascending==True and self.Is_descending==False:
            delta_pitch, delta_yaw = self.PID(self.rocket_state_pitch[0], self.rocket_state_yaw[0])
            #print(delta_pitch*180/np.pi)
            #time.sleep(1)
            self.pitch_canard.commandCanard(delta_pitch)
            self.yaw_canard.commandCanard(delta_yaw)
            self.environment.dtcnt = 0
        elif self.Is_descending==False:
            self.pitch_canard.rotateCanard()
            self.yaw_canard.rotateCanard()
            if self.Is_ascending==True:
                self.environment.dtcnt+=1
        
        # 로켓 log to csv
        self.canard_angle_log.append(self.pitch_canard.Canard_angle*180/np.pi)

        self.pitch_pid_log.append(self.pitch_canard.Canard_angle*180/np.pi)
        self.yaw_pid_log.append(self.yaw_canard.Canard_angle)
        return


    def simulate(self):
        t = np.linspace(0, 15, int(15 / self.environment.dt))
        time_range = []
        while self.simulation_flag:
            self.updateEngineState()
            self.updateCanard()
            self.Force()
            self.Translational()
            self.Rotational()
            self.environment.updateT()
            if self.environment.current_time - self.ascending_time > 6:
                self.Is_descending = True
            time_range.append(self.environment.current_time)
            if ((self.coordinate[2] <= 0 and self.environment.current_time >= 2)or(self.rocket_vel_array[2]<0)):
                self.simulation_flag = False
        

        #print(self.abs_integral_yaw)
        #plt.plot(time_range, self.pitch_canard.alpha_log)
        #plt.plot(time_range, self.yaw_torque_log)
        #plt.show()

        #plt.plot(time_range[:-1000], self.pitch_torque_log[:-1000])
        #plt.plot(time_range[:-1000], self.yaw_torque_log[:-1000])
        #plt.plot(time_range[:-2], self.pitch_angle_log[:-2])
        #plt.plot(time_range[:-1000], self.total_torque_log[:-1000])
        #plt.plot(time_range, self.thrust_log)
        #plt.show()

        #plt.plot(time_range, self.z_log)
        #plt.plot(time_range, self.velocity_z)
        #plt.plot(time_range, self.thrust_log)
        #plt.show()

        data = {
            'Time': time_range,
            'Rocket head': self.dir,
            'Rocket Angle': self.rocket_angle_log,
            'Wind Vector': self.windvector_log,
            'Canard angle': self.canard_angle_log,
            'Total Force' : self.force_log,
            'Acc' : self.accelaration_csv_log,
            'Rocket Speed': self.rocekt_velocity_log,
            'Rocket Angular Speed': self.rocekt_angular_velocity_log,
            'Canard final wind': self.pitch_canard.relative_final_log,
            'Canard Relative Wind Speed': self.canard_relative_wind_log ,
            'Final wind': self.final_wind_log,
            'AOA': self.aoa_log,
            'Canard Lift': self.lift_log,
            'Tail Lift': self.tail_lift_log,
            'CG:':self.cg_log,
            'Torque:': self.torque_log,
            'Altitude':self.z_log
        }

        cmp_data = {
            'time':time_range,
            'alt':self.cmp_alt,
            'vel':self.cmp_vel,
            'acc':self.cmp_acc
        }
        

        #df_cmp = pd.DataFrame(cmp_data)
        #df_cmp.to_csv("cmp_log.csv", index=False)

        #df = pd.DataFrame(data)

        # CSV 파일로 저장
        #df.to_csv('rocket_data_no1.csv', index=False)

        #print("CSV 파일이 성공적으로 저장되었습니다.")
        return {
            'time_range': time_range,
            'x': self.x_log,
            'y': self.y_log,
            'z': self.z_log,

            'pitch_angle_log': self.pitch_angle_log,
            'yaw_angle_log': self.yaw_angle_log,

            'thrust': self.thrust_log,

            'velocity_x': self.velocity_x,
            'velocity_y': self.velocity_y,
            'velocity_z': self.velocity_z,

            'pitch_pid_log' : self.pitch_angle_log,
            'yaw_pid_log' : self.yaw_angle_log,

            'pitch_angluar_velocity' : self.pitch_angluar_velocity_log,
            'yaw_angluar_velocity' : self.yaw_angluar_velocity_log
        }
    
    def position(self):
        return self.pos, self.dir

"""rocket = Rocket(0.661,0.954,3.92,0.38,0,0,1.146174346,[-3,2,-0.3],0.001,0.26,0.0018,"","",0.2617993333,1.2,0.018,"","",0.99,0.4,0.28,0.2,0.02,0.01,
"rocket flight simulator/data/thrust.csv","rocket flight simulator/data/pressure.csv",0.5,0.3,0.01,0.052,2)

print("객체 생성 완료")


rocket.simulate()
print("시뮬레이션 완료")"""

"""
coordinates, directions = rocket.position()
print(coordinates[-1][0], coordinates[-1][1], coordinates[-1][2])
coordinates = coordinates[::100]
directions = directions[::100]
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

# 벡터의 초기값
quiver = ax.quiver(0, 0, 0, 0, 0, 0)

def update(i):
    ax.cla()  # 이전 프레임 삭제
    ax.set_xlim([-100, 100])
    ax.set_ylim([-100, 100])
    ax.set_zlim([0, 200])
    # 현재 좌표와 방향으로 벡터 업데이트

    quiver = ax.quiver(coordinates[i][0], coordinates[i][1], coordinates[i][2],
                       directions[i][0]*20, directions[i][1]*20, directions[i][2]*20, color='b')
    return quiver

# 애니메이션 설정
ani = FuncAnimation(fig, update, frames=len(coordinates), interval=0, blit=False)

plt.show()
"""