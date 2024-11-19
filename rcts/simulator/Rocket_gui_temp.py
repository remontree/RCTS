import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QVBoxLayout, QPushButton, QLineEdit, QLabel, QFileDialog, QScrollArea, QFormLayout, QAction, QMenuBar
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtTest import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation
import numpy as np
from rocket import Rocket
import json
import tunner
import random
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import matplotlib.pyplot as plt
import os
import numpy as np
import time

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=7, height=5, dpi=100, projection=None):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111, projection=projection) if projection else self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("RCTS-I")
        self.setGeometry(100, 100, 1600, 1000)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        grid_layout = QGridLayout()
        central_widget.setLayout(grid_layout)

        # 메뉴바 생성
        menubar = self.menuBar()
        file_menu = menubar.addMenu('File')

        save_action = QAction('Save', self)
        save_action.triggered.connect(self.save_params)
        file_menu.addAction(save_action)

        load_action = QAction('Load', self)
        load_action.triggered.connect(self.load_params)
        file_menu.addAction(load_action)

        # 그래프 화면
        self.plot1 = MplCanvas(self, width=7, height=5, dpi=100, projection='3d')
        self.plot2 = MplCanvas(self, width=7, height=5, dpi=100)
        self.plot3 = MplCanvas(self, width=7, height=5, dpi=100)
        self.plot4 = MplCanvas(self, width=7, height=5, dpi=100)
        self.plot5 = MplCanvas(self, width=7, height=5, dpi=100)
        self.plot6 = MplCanvas(self, width=7, height=5, dpi=100)

        grid_layout.addWidget(self.plot1, 0, 0)
        grid_layout.addWidget(self.plot2, 0, 1)
        grid_layout.addWidget(self.plot3, 0, 2)
        grid_layout.addWidget(self.plot4, 1, 0)
        grid_layout.addWidget(self.plot5, 1, 1)
        grid_layout.addWidget(self.plot6, 1, 2)

        # 파라미터 입력창
        param_layout = QFormLayout()
        self.param_widget = QWidget()
        self.param_widget.setLayout(param_layout)

        self.add_param_input(param_layout, 'cg', '질량 중심 (m)')
        self.add_param_input(param_layout, 'cp', '압력 중심 (m)')
        self.add_param_input(param_layout, 'mass', '질량 (kg)')
        self.add_param_input(param_layout, 'I', '관성 모멘트 (kg·m²)')
        self.add_param_input(param_layout, 'pitch_init_angle', '초기 피치 각도 (rad)')
        self.add_param_input(param_layout, 'yaw_init_angle', '초기 요 각도 (rad)')
        self.add_param_input(param_layout, 'air_density', '공기 밀도 (kg/m³)')
        self.add_param_input(param_layout, 'wind_velocity', '바람 속도 (m/s)')
        self.add_param_input(param_layout, 'dt', '시간 간격 (s)')
        self.add_param_input(param_layout, 'canard_x', '카나드 위치 (m)')
        self.add_param_input(param_layout, 'canard_area', '카나드 면적 (m²)')
        self.add_param_input(param_layout, 'Cl_canard_path', '카나드 양력계수 경로', is_path=True)
        self.add_param_input(param_layout, 'Cd_canard_path', '카나드 항력계수 경로', is_path=True)
        self.add_param_input(param_layout, 'Canard_Max_Angle', '카나드 최대 각도 (rad)')
        self.add_param_input(param_layout, 'tail_x', '꼬리 위치 (m)')
        self.add_param_input(param_layout, 'tail_area', '꼬리 면적 (m²)')
        self.add_param_input(param_layout, 'Cl_tail_path', '꼬리 양력계수 경로', is_path=True)
        self.add_param_input(param_layout, 'Cd_tail_path', '꼬리 항력계수 경로', is_path=True)
        self.add_param_input(param_layout, 'x', '로켓 위치 (m)')
        self.add_param_input(param_layout, 'fuel_M', '연료 질량 (kg)')
        self.add_param_input(param_layout, 'fuel_M_F', '연료 유량 (kg/s)')
        self.add_param_input(param_layout, 'fuel_length', '연료 길이 (m)')
        self.add_param_input(param_layout, 'fuel_r1', '연료 반지름1 (m)')
        self.add_param_input(param_layout, 'fuel_r2', '연료 반지름2 (m)')
        self.add_param_input(param_layout, 'thrust_data_path', '추력 데이터 경로', is_path=True)
        self.add_param_input(param_layout, 'pressure_data_path', '압력 데이터 경로', is_path=True)
        self.add_param_input(param_layout, 'kp', 'PID 제어: Kp')
        self.add_param_input(param_layout, 'ki', 'PID 제어: Ki')
        self.add_param_input(param_layout, 'kd', 'PID 제어: Kd')
        self.add_param_input(param_layout, 'R', '로켓 반지름 (m)')
        self.add_param_input(param_layout, 'Launcher_length', '발사대 길이 (m)')

        self.simulate_button = QPushButton('시뮬레이션')
        self.simulate_button.clicked.connect(self.simulate)
        param_layout.addRow(self.simulate_button, QLabel())

        self.tune_button = QPushButton('튜닝')
        self.tune_button.clicked.connect(self.tune)
        param_layout.addRow(self.tune_button, QLabel())

        scroll = QScrollArea()
        scroll.setWidget(self.param_widget)
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(350)

        grid_layout.addWidget(scroll, 0, 3, 2, 1)

        self.cnt_simulation = 0

    def add_param_input(self, layout, param_name, label_text, is_path=False):
        label = QLabel(label_text)
        line_edit = QLineEdit()
        line_edit.setObjectName(param_name)
        line_edit.setFixedWidth(300)
        layout.addRow(label, line_edit)
        if is_path:
            line_edit.setPlaceholderText('파일 경로를 입력하세요.')

    def get_param(self, param_name, is_array=False, is_path=False):
        line_edit = self.param_widget.findChild(QLineEdit, param_name)
        text = line_edit.text().strip()
        if is_path:
            return text 
        if is_array:
            return list(map(float, text.split(',')))
        return float(text) if text else None 
    
    def save_params(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save Parameters", "", "JSON Files (*.json);;All Files (*)", options=options)
        if file_name:
            try:
                params = {}
                for widget in self.param_widget.findChildren(QLineEdit):
                    params[widget.objectName()] = widget.text()
                with open(file_name, 'w') as file:
                    json.dump(params, file, indent=4)
                print("Parameters saved successfully.")
            except Exception as e:
                print(f"Error saving parameters: {e}")

    def load_params(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Load Parameters", "", "JSON Files (*.json);;All Files (*)", options=options)
        if file_name:
            try:
                with open(file_name, 'r') as file:
                    params = json.load(file)
                for widget in self.param_widget.findChildren(QLineEdit):
                    if widget.objectName() in params:
                        widget.setText(params[widget.objectName()])
                print("Parameters loaded successfully.")
            except Exception as e:
                print(f"Error loading parameters: {e}")


    def simulate(self):
        popup = QMessageBox()
        popup.setWindowTitle("Simulation in Progress")
        popup.setText("<span style='color:blue;'>계산 중입니다. 잠시만 기다려주세요.</span>")
        popup.setStandardButtons(QMessageBox.NoButton)  # No buttons to prevent interruption
        popup.show()

        # Force the GUI to update and show the popup immediately
        QApplication.processEvents()

        if self.cnt_simulation == 0:
            try:
                self.rocket = Rocket(
                    self.get_param('cg'), self.get_param('cp'), self.get_param('mass'), self.get_param('I'),
                    self.get_param('pitch_init_angle'), self.get_param('yaw_init_angle'), self.get_param('air_density'),
                    self.get_param('wind_velocity', is_array=True), self.get_param('dt'), self.get_param('canard_x'),
                    self.get_param('canard_area'), self.get_param('Cl_canard_path', is_path=True), 
                    self.get_param('Cd_canard_path', is_path=True), self.get_param('Canard_Max_Angle'),
                    self.get_param('tail_x'), self.get_param('tail_area'), self.get_param('Cl_tail_path', is_path=True),
                    self.get_param('Cd_tail_path', is_path=True), self.get_param('x'), self.get_param('fuel_M'),
                    self.get_param('fuel_M_F'), self.get_param('fuel_length'), self.get_param('fuel_r1'),
                    self.get_param('fuel_r2'), self.get_param('thrust_data_path', is_path=True),
                    self.get_param('pressure_data_path', is_path=True), self.get_param('kp'), self.get_param('ki'),
                    self.get_param('kd'), self.get_param('R'), self.get_param('Launcher_length')
                )
                self.cnt_simulation += 1
            except Exception as e:
                print(f"Error in input parameters: {e}")
                return
        else:
            self.rocket.parameterUpdate(
                self.get_param('cg'), self.get_param('cp'), self.get_param('mass'), self.get_param('I'),
                self.get_param('pitch_init_angle'), self.get_param('yaw_init_angle'), self.get_param('air_density'),
                self.get_param('wind_velocity', is_array=True), self.get_param('dt'), self.get_param('canard_x'),
                self.get_param('canard_area'), self.get_param('Cl_canard_path', is_path=True), 
                self.get_param('Cd_canard_path', is_path=True), self.get_param('Canard_Max_Angle'),
                self.get_param('tail_x'), self.get_param('tail_area'), self.get_param('Cl_tail_path', is_path=True),
                self.get_param('Cd_tail_path', is_path=True), self.get_param('x'), self.get_param('fuel_M'),
                self.get_param('fuel_M_F'), self.get_param('fuel_length'), self.get_param('fuel_r1'),
                self.get_param('fuel_r2'), self.get_param('thrust_data_path', is_path=True),
                self.get_param('pressure_data_path', is_path=True), self.get_param('kp'), self.get_param('ki'),
                self.get_param('kd'), self.get_param('R'), self.get_param('Launcher_length')
            )
            self.cnt_simulation += 1

        data = self.rocket.simulate()

        popup.close()

        coordinates, directions = self.rocket.position()

        coordinates = coordinates[::100]
        directions = directions[::100]

        def update(i):
            self.plot1.axes.cla()  # 이전 프레임 삭제
            self.plot1.axes.set_xlim([-200, 200])
            self.plot1.axes.set_ylim([-200, 200])
            self.plot1.axes.set_zlim([0, 200])
            # 현재 좌표와 방향으로 벡터 업데이트
            self.plot1.axes.quiver(coordinates[i][0], coordinates[i][1], coordinates[i][2],
                                directions[i][0] * 20, directions[i][1] * 20, directions[i][2] * 20, color='b')

        # 애니메이션 객체를 인스턴스 변수로 유지
        self.ani = FuncAnimation(self.plot1.fig, update, frames=len(coordinates), interval=50, blit=False)

        self.plot1.draw()


        # 비행 궤적
        """self.plot1.axes.clear()
        self.plot1.axes.plot(data['x'], data['y'], data['z'])
        self.plot1.axes.set_xlabel('X axis')
        self.plot1.axes.set_ylabel('Y axis')
        self.plot1.axes.set_zlabel('Z axis')
        self.plot1.axes.set_xlim(-200,200)
        self.plot1.axes.set_ylim(-200,200)
        self.plot1.draw()"""

        # Pitch yaw 각도에 대한 PID 제어 현황
        self.plot2.axes.clear()
        self.plot2.axes.plot(data['time_range'], data['pitch_angle_log'], label='Pitch angle')
        self.plot2.axes.plot(data['time_range'], data['yaw_angle_log'], label='Yaw angle')
        self.plot2.axes.axhline(y=0, color='r', linestyle='--', label='Desired angle')
        self.plot2.axes.set_xlabel('Time (s)')
        self.plot2.axes.set_ylabel('Angle (rad)')
        self.plot2.axes.legend()
        self.plot2.draw()

        # 고도 변화
        self.plot3.axes.clear()
        self.plot3.axes.plot(data['time_range'], data['z'], label='Altitude')
        self.plot3.axes.set_xlabel('Time (s)')
        self.plot3.axes.set_ylabel('Altitude (m)')
        self.plot3.axes.legend()
        self.plot3.draw()

        # CP(center of pressure)에 걸리는 힘
        self.plot4.axes.clear()
        self.plot4.axes.plot(data['time_range'], data['thrust'], label='total force')
        self.plot4.axes.set_xlabel('Time (s)')
        self.plot4.axes.set_ylabel('Force (N)')
        self.plot4.axes.legend()
        self.plot4.draw()

        # Pitch yaw의 카나드 변화
        self.plot5.axes.clear()
        self.plot5.axes.plot(data['time_range'], data['pitch_angluar_velocity'], label='Pitch angular velocity')
        self.plot5.axes.plot(data['time_range'], data['yaw_angluar_velocity'], label='Yaw angular velocity')
        self.plot5.axes.set_xlabel('Time (s)')
        self.plot5.axes.set_ylabel('pid change (rad)')
        self.plot5.axes.legend()
        self.plot5.draw()

        # Rocket velocity
        self.plot6.axes.clear()
        self.plot6.axes.plot(data['time_range'], data['velocity_x'], label='x velocity')
        self.plot6.axes.plot(data['time_range'], data['velocity_y'], label='y velocity')
        self.plot6.axes.plot(data['time_range'], data['velocity_z'], label='z velocity')
        self.plot6.axes.set_xlabel('Time (s)')
        self.plot6.axes.set_ylabel('Velocity (m/s)')
        self.plot6.axes.legend()
        self.plot6.draw()

    def tune(self):
        try:
            rocket = tunner.RocketEnv(self.rocket)
        except Exception as e:
            print(f"Error in input parameters: {e}")
            return


        agent = tunner.DQNAgent()
        score_history = []

        agent.model.load_state_dict(torch.load("/home/remon/문서/rcts/rcts/simulator/model_episode_1000.pth"))

        state = rocket.reset()[0]

        while True:
            popup = QMessageBox()
            popup.setWindowTitle("Simulation in Progress")
            popup.setText("<span style='color:blue;'>계산 중입니다. 잠시만 기다려주세요.</span>")
            popup.setStandardButtons(QMessageBox.NoButton)  # No buttons to prevent interruption
            popup.show()
            state_tensor = torch.FloatTensor([state]).to(tunner.device)
            action = agent.act(state_tensor)
            popup.close()
            next_state, reward, done, info = rocket.step(action.item())
            
            agent.memorize(state_tensor, action, reward, next_state)
            #agent.learn()

            state = next_state

            self.param_widget.findChild(QLineEdit, 'kp').setText(str(rocket.rocket.kp))
            self.param_widget.findChild(QLineEdit, 'ki').setText(str(rocket.rocket.ki))
            self.param_widget.findChild(QLineEdit, 'kd').setText(str(rocket.rocket.kd))


            data = rocket.data

            coordinates, directions = self.rocket.position()

            coordinates = coordinates[::100]
            directions = directions[::100]

            def update(i):
                self.plot1.axes.cla()  # 이전 프레임 삭제
                self.plot1.axes.set_xlim([-200, 200])
                self.plot1.axes.set_ylim([-200, 200])
                self.plot1.axes.set_zlim([0, 200])
                # 현재 좌표와 방향으로 벡터 업데이트
                self.plot1.axes.quiver(coordinates[i][0], coordinates[i][1], coordinates[i][2],
                                    directions[i][0] * 20, directions[i][1] * 20, directions[i][2] * 20, color='b')

            # 애니메이션 객체를 인스턴스 변수로 유지
            self.ani = FuncAnimation(self.plot1.fig, update, frames=len(coordinates), interval=50, blit=False)

            self.plot1.draw()

            print("wow")


            # 비행 궤적
            """self.plot1.axes.clear()
            self.plot1.axes.plot(data['x'], data['y'], data['z'])
            self.plot1.axes.set_xlabel('X axis')
            self.plot1.axes.set_ylabel('Y axis')
            self.plot1.axes.set_zlabel('Z axis')
            self.plot1.axes.set_xlim(-200,200)
            self.plot1.axes.set_ylim(-200,200)
            self.plot1.draw()"""

            # Pitch yaw 각도에 대한 PID 제어 현황
            self.plot2.axes.clear()
            self.plot2.axes.plot(data['time_range'], data['pitch_angle_log'], label='Pitch angle')
            self.plot2.axes.plot(data['time_range'], data['yaw_angle_log'], label='Yaw angle')
            self.plot2.axes.axhline(y=0, color='r', linestyle='--', label='Desired angle')
            self.plot2.axes.set_xlabel('Time (s)')
            self.plot2.axes.set_ylabel('Angle (rad)')
            self.plot2.axes.legend()
            self.plot2.draw()

            # 고도 변화
            self.plot3.axes.clear()
            self.plot3.axes.plot(data['time_range'], data['z'], label='Altitude')
            self.plot3.axes.set_xlabel('Time (s)')
            self.plot3.axes.set_ylabel('Altitude (m)')
            self.plot3.axes.legend()
            self.plot3.draw()

            # CP(center of pressure)에 걸리는 힘
            self.plot4.axes.clear()
            self.plot4.axes.plot(data['time_range'], data['thrust'], label='total force')
            self.plot4.axes.set_xlabel('Time (s)')
            self.plot4.axes.set_ylabel('Force (N)')
            self.plot4.axes.legend()
            self.plot4.draw()

            # Pitch yaw의 카나드 변화
            self.plot5.axes.clear()
            self.plot5.axes.plot(data['time_range'], data['pitch_angluar_velocity'], label='Pitch angular velocity')
            self.plot5.axes.plot(data['time_range'], data['yaw_angluar_velocity'], label='Yaw angular velocity')
            self.plot5.axes.set_xlabel('Time (s)')
            self.plot5.axes.set_ylabel('pid change (rad)')
            self.plot5.axes.legend()
            self.plot5.draw()

            # Rocket velocity
            self.plot6.axes.clear()
            self.plot6.axes.plot(data['time_range'], data['velocity_x'], label='x velocity')
            self.plot6.axes.plot(data['time_range'], data['velocity_y'], label='y velocity')
            self.plot6.axes.plot(data['time_range'], data['velocity_z'], label='z velocity')
            self.plot6.axes.set_xlabel('Time (s)')
            self.plot6.axes.set_ylabel('Velocity (m/s)')
            self.plot6.axes.legend()
            self.plot6.draw()

            
            QTest.qWait(1000)

            if done:
                print("Score:{0}".format(reward))
                score_history.append(reward)
                break

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
