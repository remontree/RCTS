import pandas as pd
import segment
import numpy as np
import time
import math

class Engine(segment.Segment):
    fuel_M = None # 연료 초기 질량
    fuel_M_F = None # 연료 최종 질량
    current_fuel_mass = None # 현재 연료 질량
    fuel_I = None # 연료 관성 모멘트
    fuel_length = None # 연료 길이
    fuel_r1 = None # 연료 외경
    fuel_r2 = None # 연료 초기 내경
    current_fuel_r2 = None # 현재 연료 내경
    fuel_density = None # 연료 밀도

    dt = None # 미소 시간

    thrust_data = None # 추력 TMS 데  이터 
    pressure_data = None # 압력 TMS 데이터

    pressure_integral = None # 압력 적분 값
    pressure_area = None # 압력 0에서 마지막까지의 정적분 값

    thrust = None # 추력
    pressure = None # 압력

    c1 = None # 연료의 질량 변화와 압력 사이의 비율
    c2 = None # t초 후의 질량을 구하는 미분방정식의 적분 상수
    c3 = None # t초 후의 반지름을 구하는 미분방정식의 적분 상수

    def __init__(self, x, fuel_M, fuel_M_F, fuel_length, fuel_r1, fuel_r2, dt, thrust_data_path, pressure_data_path):
        self.d = x
        self.fuel_M = fuel_M
        self.fuel_M_F = fuel_M_F
        self.current_fuel_mass = fuel_M
        self.fuel_length = fuel_length
        self.fuel_r1 = fuel_r1
        self.fuel_r2 = fuel_r2
        self.current_fuel_r2 = fuel_r2 
        self.thrust_data = pd.read_csv(thrust_data_path, sep = ",")
        self.thrust_data.set_index('Time', inplace=True)
        self.thrust_data_dictionary = {}
        self.pressure_data = pd. read_csv(pressure_data_path, sep = ",")
        self.pressure_data.set_index('Time', inplace=True)
        self.pressure_data_dictionary = {}
        self.pressure_integral = 0    
        self.pressure_area = 0
        self.thrust = 0
        self.pressure = 0
        self.dt = dt
        self.fuel_density = self.fuel_M/(math.pi*(self.fuel_r1**2-self.fuel_r2**2)*self.fuel_length)
        
        self.initPressureThrust()
        self.getC_1_2()
        self.updateI()

        self.function_cnt = 0
        self.mean = 0

    def updateparam(self,x, fuel_M, fuel_M_F, fuel_length, fuel_r1, fuel_r2, dt, thrust_data_path, pressure_data_path):
        self.d = x
        self.fuel_M = fuel_M
        self.fuel_M_F = fuel_M_F
        self.current_fuel_mass = fuel_M
        self.fuel_length = fuel_length
        self.fuel_r1 = fuel_r1
        self.fuel_r2 = fuel_r2
        self.current_fuel_r2 = fuel_r2 
        self.pressure_integral = 0    
        self.pressure_area = 0
        self.thrust = 0
        self.pressure = 0
        self.dt = dt
        self.fuel_density = self.fuel_M/(math.pi*(self.fuel_r1**2-self.fuel_r2**2)*self.fuel_length)
        self.updateI()
    # 압력 데이터 초기화
    def initPressureThrust(self):
        t = 0
        while self.pressure_data.index[-1]>t:
            self.pressure_data = self.pressure_data.reindex(self.pressure_data.index.union([t]))
            self.pressure_data['Pressure'] = self.pressure_data['Pressure'].interpolate()
            pressure = self.pressure_data['Pressure'][t]
            self.pressure_data_dictionary[t] = pressure
            self.pressure_area += pressure*self.dt

            self.thrust_data = self.thrust_data.reindex(self.thrust_data.index.union([t]))
            self.thrust_data['Thrust'] = self.thrust_data['Thrust'].interpolate()
            self.thrust_data_dictionary[t] = self.thrust_data.loc[t, 'Thrust']

            t+=self.dt

    # c1과 c2 구하는 함수
    def getC_1_2(self):
        self.c2 = self.fuel_M
        self.c1 = (self.fuel_M_F - self.c2)/self.pressure_area
    
    # 현재 압력 초기화
    def updatePressure(self, env):
        if self.pressure_data.index[-1]<env.current_time:
            self.pressure = 0
        else:
            try:
                #self.pressure = self.pressure_data['Pressure'][env.current_time]
                self.pressure = self.pressure_data_dictionary[env.current_time]
            except:
                print("pressure execption occured!")
                self.pressure = self.pressure_data.reindex(self.pressure_data.index.union([env.current_time]))
                self.pressure_data['Pressure'] = self.pressure_data['Pressure'].interpolate()
                self.pressure = self.pressure_data['Pressure'][env.current_time]
        #self.pressure = 100
        self.pressure_integral += self.pressure*env.dt

    # 현재 질량 업데이트
    def updateMass(self):
        self.current_fuel_mass = self.c1*self.pressure_integral+self.c2

    # 현재 추력 업데이트
    def updateThrust(self, env):
        if self.thrust_data.index[-1]<env.current_time:
            self.thrust = 0
        else:
            try:
                self.thrust = self.thrust_data_dictionary[env.current_time]
            except:
                print("thrust execption occured!")
                self.thrust_data = self.thrust_data.reindex(self.thrust_data.index.union([env.current_time]))
                self.thrust_data['Thrust'] = self.thrust_data['Thrust'].interpolate()
                self.thrust = self.thrust_data.loc[env.current_time, 'Thrust']

    # 이건 조금 고려해 볼 필요가 있을 듯
    def updateR2(self):
        return -1
    
    # 관성 모멘트 업데이트
    def updateI(self):
        self.fuel_I = (1/4)*self.current_fuel_mass*(self.fuel_r1**2+self.current_fuel_r2**2)

    # 현재 엔진 상태 한번에 업데이트
    def updateEngineState(self, environment):
        self.updatePressure(environment)
        self.updateMass()
        self.updateThrust(environment)
        self.updateR2()
        self.updateI()
    

