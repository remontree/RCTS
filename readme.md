# rocket and missile flight simulation software to solve optimal pid coefficient with reinforcement learning model and Ziegler Nichols Method

this software simulate rocket's flight to design optimal rocket and solve optimal pid coefficient for canard fin control.

this simulation simulates rocket's rotationrotational movement and translational movement.

## feature
1. simulate rocket's flight
2. solve optimal pid coefficient for canard fin control with reinforcement learnging and Ziegler Nichols Method

## algoritm<br>
1. Changes in the rocket’s center of mass and moment of inertia due to fuel combustion

2. Relative velocity vector and aoa (angle of attack) of the fluid entering each canard fin, tail rotor, and body.

3. Calculate the lift and drag force received by each wing and body according to the relative velocity vector and aoa of the fluid.

4. Net force calculation that comprehensively calculates the thrust, gravity, lift, and drag of the rocket

5. Calculate the torque received by each part of the rocket

6. Calculate the rotational and translational motion of the rocket over time

7. Calculate the output value of the PID controller and canard pin state change according to the rocket’s attitude.

## required parameters<br>

<div class="table-overflow"><div class="table-overflow"><table style="border-collapse: collapse; width: 100.93%; height: 417px;" border="1" data-ke-align="alignLeft" data-ke-style="style5">
<tbody>
<tr style="height: 20px;">
<td style="width: 18.7861%; height: 20px;"><span><span>번호</span></span></td>
<td style="width: 231.792%; height: 20px;"><span><span>매개변수</span></span></td>
</tr>
<tr style="height: 20px;">
<td style="width: 18.7861%; height: 20px;"><span><span>1</span></span></td>
<td style="width: 231.792%; height: 20px;"><span><span>질량 중심</span></span></td>
</tr>
<tr style="height: 20px;">
<td style="width: 18.7861%; height: 20px;"><span><span>2</span></span></td>
<td style="width: 231.792%; height: 20px;"><span><span>압력 중심</span></span></td>
</tr>
<tr style="height: 20px;">
<td style="width: 18.7861%; height: 20px;"><span><span>3</span></span></td>
<td style="width: 231.792%; height: 20px;"><span><span>질량</span></span></td>
</tr>
<tr style="height: 20px;">
<td style="width: 18.7861%; height: 20px;"><span><span>4</span></span></td>
<td style="width: 231.792%; height: 20px;"><span><span>관성 모멘트</span></span></td>
</tr>
<tr style="height: 20px;">
<td style="width: 18.7861%; height: 20px;"><span><span>5</span></span></td>
<td style="width: 231.792%; height: 20px;"><span><span>초기 피치</span><span>, </span><span>요 각도</span></span></td>
</tr>
<tr style="height: 20px;">
<td style="width: 18.7861%; height: 20px;"><span><span>6</span></span></td>
<td style="width: 231.792%; height: 20px;"><span><span>공기 밀도</span></span></td>
</tr>
<tr style="height: 20px;">
<td style="width: 18.7861%; height: 20px;"><span><span>7</span></span></td>
<td style="width: 231.792%; height: 20px;"><span><span>바람 속도</span></span></td>
</tr>
<tr style="height: 20px;">
<td style="width: 18.7861%; height: 20px;"><span><span>8</span></span></td>
<td style="width: 231.792%; height: 20px;"><span><span>시간 간격</span></span></td>
</tr>
<tr style="height: 20px;">
<td style="width: 18.7861%; height: 20px;"><span><span>9</span></span></td>
<td style="width: 231.792%; height: 20px;"><span><span>카나드 위치</span><span>, </span><span>면적</span></span></td>
</tr>
<tr style="height: 40px;">
<td style="width: 18.7861%; height: 40px;"><span><span>10</span></span></td>
<td style="width: 231.792%; height: 40px;"><span><span>카나드</span><span>, </span><span>꼬리날개의 양력</span><span>, </span><span>항력 계수 </span><span>cfd </span><span>데이터</span></span></td>
</tr>
<tr style="height: 20px;">
<td style="width: 18.7861%; height: 20px;"><span><span>11</span></span></td>
<td style="width: 231.792%; height: 20px;"><span><span>카나드 최대 각도</span></span></td>
</tr>
<tr style="height: 20px;">
<td style="width: 18.7861%; height: 20px;"><span><span>12</span></span></td>
<td style="width: 231.792%; height: 20px;"><span><span>꼬리날개 위치</span><span>, </span><span>면적</span></span></td>
</tr>
<tr style="height: 20px;">
<td style="width: 18.7861%; height: 20px;"><span><span>13</span></span></td>
<td style="width: 231.792%; height: 20px;"><span><span>로켓 길이</span></span></td>
</tr>
<tr style="height: 20px;">
<td style="width: 18.7861%; height: 20px;"><span><span>14</span></span></td>
<td style="width: 231.792%; height: 20px;"><span><span>연료 질량</span><span>, </span><span>유량</span><span>, </span><span>내경</span><span>, </span><span>외경</span></span></td>
</tr>
<tr style="height: 20px;">
<td style="width: 18.7861%; height: 20px;"><span><span>15</span></span></td>
<td style="width: 231.792%; height: 20px;"><span><span>엔진 추력 데이터</span></span></td>
</tr>
<tr style="height: 20px;">
<td style="width: 18.7861%; height: 20px;"><span><span>16</span></span></td>
<td style="width: 231.792%; height: 20px;"><span><span>엔진 압력 데이터</span></span></td>
</tr>
<tr style="height: 17px;">
<td style="width: 18.7861%; height: 17px;"><span><span>17</span></span></td>
<td style="width: 231.792%; height: 17px;"><span><span>Kp, Ki, Kd</span></span></td>
</tr>
<tr style="height: 20px;">
<td style="width: 18.7861%; height: 20px;"><span><span>18</span></span></td>
<td style="width: 231.792%; height: 20px;"><span><span>로켓 외경</span></span></td>
</tr>
<tr style="height: 20px;">
<td style="width: 18.7861%; height: 20px;"><span><span>19</span></span></td>
<td style="width: 231.792%; text-align: left; height: 20px;"><span><span>발사대 길이</span></span></td>
</tr>
</tbody>
</table></div></div>

## software picture
<img src = "src/software.png">

