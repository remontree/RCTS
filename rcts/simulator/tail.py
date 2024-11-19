from fin import Fin
from environment import Environment
import numpy as np
import math

class Tail(Fin):
    def __init__(self, x, Area, Cl_path, Cd_path, axis):
        super().__init__(x, Area, Cl_path, Cd_path, axis)


        #데이터 to csv

        self.sign_log = []

    def compute_Alpha(self, axis_vector, pitch_state, yaw_state, rocket_velocity, environment, d):
        super().compute_Alpha(axis_vector, pitch_state, yaw_state, rocket_velocity, environment, d)

        if self.axis == "yaw":
            relative_velocity_T = self.relative_velocity[:]

            relative_velocity_norm = np.linalg.norm(np.array(relative_velocity_T))

            # 로켓 헤드 방향과 relative_velocity 사이의 각도 구하기 (최적화 해서 일부 연산 축약함)
            dot_vector = relative_velocity_T[2]

            self.Alpha = math.acos(dot_vector/(relative_velocity_norm))

            if self.Alpha*180/np.pi > 90:
                self.Alpha = np.pi-self.Alpha
        
        elif self.axis == "pitch":
            relative_velocity_T = self.relative_velocity[:]

            relative_velocity_norm = np.linalg.norm(np.array(relative_velocity_T))

            # 로켓 헤드 방향과 relative_velocity 사이의 각도 구하기 (최적화 해서 일부 연산 축약함)
            dot_vector = relative_velocity_T[2]

            self.Alpha = math.acos(dot_vector/(relative_velocity_norm))

            if self.Alpha*180/np.pi > 90:
                self.Alpha = np.pi-self.Alpha

        # 양력 방향 결정
        if self.axis == "pitch":
            if relative_velocity_T[1]>0:
                self.lift_direction = 1
            else:
                self.lift_direction = -1

            # 데이터 to csv
            self.sign_log.append((relative_velocity_T[1], self.lift_direction))
        elif self.axis == "yaw":
            if relative_velocity_T[0]>0:
                self.lift_direction = -1
            else:
                self.lift_direction = 1

        return self.Alpha, self.lift_direction
    
    def compute_lift_force(self, environment):
        #lift_force = 0.5* super().compute_lift_coefficient() * environment.air_density * self.Area * (np.linalg.norm(self.relative_velocity) ** 2)
        lift_force = 0.5* super().compute_lift_coefficient() * environment.air_density * self.Area * (np.linalg.norm(self.relative_velocity) ** 2) * abs(math.cos(self.Alpha))
        lift_force = lift_force * self.lift_direction
        return lift_force
    
    def compute_drag_force(self, environment):
        drag_force = 0.5* super().compute_drag_coefficient(np.linalg.norm(self.relative_velocity)) * environment.air_density * self.Area * math.sin(self.Alpha) * (np.linalg.norm(self.relative_velocity) ** 2) * abs(math.cos(self.Alpha))
        return drag_force