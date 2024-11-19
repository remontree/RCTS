from fin import Fin
from environment import Environment
import numpy as np
import asyncio
import math

class Canard(Fin):
    def __init__(self, x, Area, Cl_path, Cd_path, Canard_Max_Angle, axis):
        super().__init__(x, Area, Cl_path, Cd_path, axis)
        self.Canard_angle = 0
        self.desired_angle = 0
        self.Canard_Max_Angle = Canard_Max_Angle
        self.alpha_log = []


    def Canard_Angle(self, angle):
        if abs(angle)>self.Canard_Max_Angle:
            if(angle<0):
                self.Canard_angle = -self.Canard_Max_Angle
            else:
                self.Canard_angle = self.Canard_Max_Angle
        else:
            self.Canard_angle = angle

    
    def commandCanard(self, desired):
        self.desired_angle = desired
    
    def rotateCanard(self):
        if self.Canard_angle>self.desired_angle:
            self.Canard_angle-=(0.33*np.pi/180)
        elif self.Canard_angle<self.desired_angle:
            self.Canard_angle+=(0.33*np.pi/180)

        if self.Canard_angle < -self.Canard_Max_Angle:
            self.Canard_angle = -self.Canard_Max_Angle
        elif self.Canard_angle > self.Canard_Max_Angle:
            self.Canard_angle = self.Canard_Max_Angle
    
    # 날개 양력 방향이랑 이런거 다시 고려해 볼 필요 있을 듯
    def compute_Alpha(self, axis_vector, pitch_state, yaw_state, rocket_velocity, environment, d):
        super().compute_Alpha(axis_vector, pitch_state, yaw_state, rocket_velocity, environment, d)
        
        if self.axis == "yaw":
            R2 = np.array([
                [math.cos(-self.Canard_angle), 0, -math.sin(-self.Canard_angle)],
                [0, 1, 0],
                [math.sin(-self.Canard_angle), 0, math.cos(-self.Canard_angle)]
            ])

            relative_velocity_T = np.dot(R2, self.relative_velocity)


            relative_velocity_norm = np.linalg.norm(np.array(relative_velocity_T))

            # 로켓 헤드 방향과 relative_velocity 사이의 각도 구하기 (최적화 해서 일부 연산 축약함)
            dot_vector = relative_velocity_T[2]

            self.Alpha = math.acos(dot_vector/(relative_velocity_norm))

            if self.Alpha*180/np.pi > 90:
                self.Alpha = np.pi-self.Alpha
        
        elif self.axis == "pitch":
            R1 = np.array([
                [1, 0, 0],
                [0, math.cos(-self.Canard_angle), math.sin(-self.Canard_angle)],
                [0, -math.sin(-self.Canard_angle), math.cos(-self.Canard_angle)]
            ])

            relative_velocity_T = np.dot(R1, self.relative_velocity)

            relative_velocity_norm = np.linalg.norm(np.array(relative_velocity_T))

            #데이터 to csv
            self.final_wind_log = relative_velocity_T

            # 로켓 헤드 방향과 relative_velocity 사이의 각도 구하기 (최적화 해서 일부 연산 축약함)
            dot_vector = relative_velocity_T[2]

            self.Alpha = math.acos(dot_vector/(relative_velocity_norm))

            if self.Alpha*180/np.pi > 90:
                self.Alpha = np.pi-self.Alpha
            
        #print("final wind: ", relative_velocity_T, "axis: ", self.axis)

        # 양력 방향 결정
        if self.axis == "pitch":
            if relative_velocity_T[1]>0:
                self.lift_direction = 1
            else:
                self.lift_direction = -1
        elif self.axis == "yaw":
            if relative_velocity_T[0]>0:
                self.lift_direction = -1
            else:
                self.lift_direction = 1

        return self.Alpha, self.lift_direction
    
    def compute_lift_force(self, environment):
        #lift_force = 0.5* super().compute_lift_coefficient() * environment.air_density * self.Area * (np.linalg.norm(self.relative_velocity) ** 2)
        lift_force = 0.5* super().compute_lift_coefficient() * environment.air_density * self.Area * (np.linalg.norm(self.relative_velocity) ** 2) * abs(math.cos(self.Alpha)) * abs(math.cos(self.Canard_angle))
        lift_force = lift_force * self.lift_direction

        return lift_force
    
    def compute_drag_force(self, environment):
        drag_force = 0.5* super().compute_drag_coefficient(np.linalg.norm(self.relative_velocity)) * environment.air_density * self.Area * math.sin(self.Alpha) * (np.linalg.norm(self.relative_velocity) ** 2) * abs(math.cos(self.Alpha))
        return drag_force