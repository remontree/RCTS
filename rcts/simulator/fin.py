from segment import Segment
from environment import Environment
import numpy as np
import pandas as pd
import math
import copy

#Fin 클래스 내의 모든 메소드는 Canard 클래스와 Tail 클래스에서 overriding
class Fin(Segment):
    #객체 속성은 초기에 설정이 필요한 값만을 추가
    def __init__(self, x, Area, Cl_path, Cd_path, axis):
        self.d = x
        self.Area = Area #단면적
        self.Alpha = 0 #Angle of Attack, 바람의 진입각
        self.lift_direction = 1 #양력방향
        #self.Lift_coef_cfd = pd.read_csv(Cl_path, sep=",") #양력계수 파일 경로
        #self.Drag_coef_cfd = pd.read_csv(Cd_path, sep=",") #항력계수 파일 경로
        self.relative_velocity = 0
        self.axis = axis


        #데이터 to csv
        self.final_wind_log = [0,0,0]
        self.relative_final_log = []

    def projection(self,v1, v2):
        dot_v1_v1 = v1[0]**2+v1[1]**2+v1[2]**2

        dot_v1_v2 = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

        scalar = dot_v1_v2/dot_v1_v1

        projection_vector = copy.deepcopy(v1)

        #projection_vector = v1[:]

        projection_vector[0]*=scalar
        projection_vector[1]*=scalar
        projection_vector[2]*=scalar

        return projection_vector

    def compute_Alpha(self, axis_vector, pitch_state, yaw_state, rocket_velocity, environment, d): #environment 객체를 파라미터로 받음
        relative_wind = [-rocket_velocity[0]+environment.wind[0], -rocket_velocity[1]+environment.wind[1], -rocket_velocity[2]+environment.wind[2]]
        #print("first wind: ", relative_wind)
        if self.axis == "yaw":
            pitch_axis_unit = np.array(axis_vector[1])
            #pitch_axis_unit = pitch_axis_unit/np.linalg.norm(pitch_axis_unit)

            # 원래 각속도는 단위 벡터에 -를 곱해야 하지만 상도속도라서 다시 -를 곱해야 함으로 그냥 둔다.
            """angluar_velocity_vector = pitch_axis_unit*yaw_state[1]*d

            for i in range(3):
                relative_wind[i] = relative_wind[i]+angluar_velocity_vector[i]"""

            axis_projected_vector = self.projection(axis_vector[2], relative_wind)
            self.relative_velocity = [relative_wind[0]-axis_projected_vector[0], relative_wind[1]-axis_projected_vector[1], relative_wind[2]-axis_projected_vector[2]]
        
        elif self.axis == "pitch":
            yaw_axis_unit = np.array(axis_vector[2])
            #yaw_axis_unit = yaw_axis_unit/np.linalg.norm(yaw_axis_unit)

            angluar_velocity_vector = yaw_axis_unit*(-pitch_state[1])*d

            for i in range(3):
                relative_wind[i] = relative_wind[i]+angluar_velocity_vector[i]

            self.relative_final_log.append(relative_wind)            
            axis_projected_vector = self.projection(axis_vector[1], relative_wind)
            self.relative_velocity = [relative_wind[0]-axis_projected_vector[0], relative_wind[1]-axis_projected_vector[1], relative_wind[2]-axis_projected_vector[2]]

        R1 = np.array([
            [1, 0, 0],
            [0, math.cos(-pitch_state[0]), math.sin(-pitch_state[0])],
            [0, -math.sin(-pitch_state[0]), math.cos(-pitch_state[0])]
        ])

        #yaw y축에 대한 회전 행렬 R2
        R2 = np.array([
                    [math.cos(-yaw_state[0]), 0, -math.sin(-yaw_state[0])],
                    [0, 1, 0],
                    [math.sin(-yaw_state[0]), 0, math.cos(-yaw_state[0])]
                ])

        self.relative_velocity = np.dot(np.dot(R2,R1), np.array(self.relative_velocity))
        #print("relative wind: ", self.relative_velocity)

    #후에 csv 파일의 형식에 따라서 수정 필요
    def compute_lift_coefficient(self):
        #Alpha_rad = math.radians(self.Alpha)
        Alpha_rad = self.Alpha
        if Alpha_rad < math.radians(13):
            lift_coef = 2 * math.pi * math.sin(Alpha_rad)
        else:
            lift_coef = math.sin(2 * Alpha_rad)
        
        return lift_coef
    
    #후에 csv 파일의 형식에 따라서 수정 필요
    def compute_drag_coefficient(self, rocket_vel):
        #Alpha_rad = math.radians(self.Alpha)
        Alpha_rad = self.Alpha
        if Alpha_rad < math.radians(13):
            if rocket_vel < 187:
                friction_coef = 1.33 / math.pow(2700 * rocket_vel, 0.5)
                drag_coef = 2 * friction_coef + 2 * math.pow(math.sin(Alpha_rad), 2)
            else:
                friction_coef = 0.074 / math.pow(2700 * rocket_vel, 1.0/5.0)
                drag_coef = 2 * friction_coef + 2 * math.pow(math.sin(Alpha_rad), 2)
        else:
            drag_coef = 2 * math.pow(math.sin(Alpha_rad), 2)
            
        return drag_coef
    
    def compute_lift_force(self):
        lift_force = None
        return lift_force
    
    def compute_drag_force(self):
        drag_force = None
        return drag_force
