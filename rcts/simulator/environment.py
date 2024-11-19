class Environment:
    def __init__(self, air_density, wind_velocity, dt):
        self.air_density = air_density
        self.wind = wind_velocity
        self.dt = dt
        self.current_time = 0
        self.g = 9.8
        self.dtcnt = 30

    def updateT(self):
        self.current_time+=self.dt

    def updatetimePass(self, time):
        self.current_time+=time

    def updateWind(self, wind_velocity):
        self.wind_velocity = wind_velocity

