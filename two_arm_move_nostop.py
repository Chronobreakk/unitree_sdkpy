import time
import sys
import threading

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

import numpy as np

kPi = 3.141592654
kPi_2 = 1.57079632

class G1JointIndex:
    # Left leg
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5

    # Right leg
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11

    WaistYaw = 12
    WaistRoll = 13        # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistA = 13           # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistPitch = 14       # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistB = 14           # NOTE: INVALID for g1 23dof/29dof with waist locked

    # Left arm
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20   # NOTE: INVALID for g1 23dof
    LeftWristYaw = 21     # NOTE: INVALID for g1 23dof

    # Right arm
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27  # NOTE: INVALID for g1 23dof
    RightWristYaw = 28    # NOTE: INVALID for g1 23dof

    kNotUsedJoint = 29 # NOTE: Weight

class Custom:
    def __init__(self):
        self.time_ = 0.0
        self.control_dt_ = 0.02  
        self.duration_ = 3.0   
        self.counter_ = 0
        self.weight = 0.
        self.weight_rate = 0.2
        self.kp = 60.
        self.kd = 1.5
        self.dq = 0.
        self.tau_ff = 0.
        self.mode_machine_ = 0
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()  
        self.low_state = None 
        self.first_update_low_state = False
        self.crc = CRC()
        self.done = False
        
        # 新增必要的变量
        self.initial_poses = None
        self.current_target_poses = None
        self.max_joint_velocity = 2.0  # rad/s
        self.monitor_angles = False
        self.last_action_index = -1
        self.lowCmdWriteThreadPtr = None
        self.stop_requested = False
        self.is_stopping = False

        self.target_pos = [
            0., kPi_2,  0., kPi_2, 0., 0., 0.,
            0., -kPi_2, 0., kPi_2, 0., 0., 0., 
            0, 0, 0
        ]

        self.arm_joints = [
          G1JointIndex.LeftShoulderPitch,  G1JointIndex.LeftShoulderRoll,
          G1JointIndex.LeftShoulderYaw,    G1JointIndex.LeftElbow,
          G1JointIndex.LeftWristRoll,      G1JointIndex.LeftWristPitch,
          G1JointIndex.LeftWristYaw,
          G1JointIndex.RightShoulderPitch, G1JointIndex.RightShoulderRoll,
          G1JointIndex.RightShoulderYaw,   G1JointIndex.RightElbow,
          G1JointIndex.RightWristRoll,     G1JointIndex.RightWristPitch,
          G1JointIndex.RightWristYaw
        #   G1JointIndex.WaistYaw,
        #   G1JointIndex.WaistRoll,
        #   G1JointIndex.WaistPitch
        ]
        self.waist_lock_pos = {
            G1JointIndex.WaistYaw: 0.0,
            G1JointIndex.WaistRoll: 0.0,
            G1JointIndex.WaistPitch: 0.0
        }

    def Init(self):
        # create publisher #
        self.arm_sdk_publisher = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self.arm_sdk_publisher.Init()

        # create subscriber # 
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

        # 定义基础姿态
        self.ready_pose = [0.06, 0.5, 0., 1.03, -1, 0.00, 0.00,    # 左臂放松姿态
                           0.06,-0.5, 0., 1.03, 1, 0.00, 0.00]     # 右臂放松姿态
        
        #wrist roll
        self.point1_pose = [0.02, 0.397, 0.251, 1.107, -1.748, 0.310, -1.723,
                            0.07, 0.009, -0.092, 0.810, 1.477, 0.221, 1.597]
# = - - = - - -
        self.point2_pose = [0.07, -0.009, 0.092, 0.810, -1.477, 0.096, -1.597, 
                            0.02, -0.397, -0.251, 1.107, 1.748, 0.22, 1.6]
        # 单次循环的时间
        self.cycle_duration = 2.0  # 每次移动4秒
        
        # 构建初始动作序列（只包含准备动作）
        self.action_sequence = [
            {"time": 0.0, "poses": "current"},
            {"time": 1, "poses": self.ready_pose},
        ]
        
        # 循环的基准时间
        self.cycle_start_time = 0.5
        self.stop_time = None  # 停止请求的时间

    def get_interpolated_pose(self, current_time):
        # 初始化时记录当前位置
        if self.initial_poses is None:
            self.initial_poses = self.get_current_joint_positions()
            # 替换"current"标记为实际位置
            for action in self.action_sequence:
                if action["poses"] == "current":
                    action["poses"] = self.initial_poses.copy()
        
        # 如果在准备阶段（前0.5秒）
        if current_time <= self.cycle_start_time:
            for i in range(len(self.action_sequence) - 1):
                start_action = self.action_sequence[i]
                end_action = self.action_sequence[i + 1]
                
                if start_action["time"] <= current_time <= end_action["time"]:
                    duration = end_action["time"] - start_action["time"]
                    if duration > 0:
                        ratio = (current_time - start_action["time"]) / duration
                        smooth_ratio = 6 * ratio**5 - 15 * ratio**4 + 10 * ratio**3
                    else:
                        smooth_ratio = 0.0
                    
                    target_poses = []
                    for j in range(len(start_action["poses"])):
                        start_pos = start_action["poses"][j]
                        end_pos = end_action["poses"][j]
                        target_pos = start_pos + smooth_ratio * (end_pos - start_pos)
                        target_poses.append(target_pos)
                    
                    return target_poses
            return self.ready_pose
        
        # 如果正在停止过程中
        if self.is_stopping and self.stop_time is not None:
            duration = self.cycle_duration * 2
            elapsed = current_time - self.stop_time
            
            if elapsed >= duration:
                return self.ready_pose
            
            ratio = elapsed / duration
            smooth_ratio = 6 * ratio**5 - 15 * ratio**4 + 10 * ratio**3
            
            # 从当前目标位置插值到准备位置
            target_poses = []
            current_pose = self.current_target_poses if self.current_target_poses else self.ready_pose
            for j in range(len(self.ready_pose)):
                start_pos = current_pose[j]
                end_pos = self.ready_pose[j]
                target_pos = start_pos + smooth_ratio * (end_pos - start_pos)
                target_poses.append(target_pos)
            
            return target_poses
        
        # 循环阶段：在point1和point2之间无限循环
        cycle_time = current_time - self.cycle_start_time
        full_cycle_duration = self.cycle_duration * 2  # 点1->点2->点1 的完整周期
        
        # 计算在当前周期内的位置
        position_in_cycle = cycle_time % full_cycle_duration
        
        if position_in_cycle < self.cycle_duration:
            # 从ready_pose到point1（第一次）或从point2到point1
            ratio = position_in_cycle / self.cycle_duration
            smooth_ratio = 6 * ratio**5 - 15 * ratio**4 + 10 * ratio**3
            
            # 判断是否是第一次循环
            if cycle_time < full_cycle_duration:
                start_pose = self.ready_pose
            else:
                start_pose = self.point2_pose
            
            target_poses = []
            for j in range(len(self.point1_pose)):
                start_pos = start_pose[j]
                end_pos = self.point1_pose[j]
                target_pos = start_pos + smooth_ratio * (end_pos - start_pos)
                target_poses.append(target_pos)
            
            return target_poses
        else:
            # 从point1到point2
            ratio = (position_in_cycle - self.cycle_duration) / self.cycle_duration
            smooth_ratio = 6 * ratio**5 - 15 * ratio**4 + 10 * ratio**3
            
            target_poses = []
            for j in range(len(self.point2_pose)):
                start_pos = self.point1_pose[j]
                end_pos = self.point2_pose[j]
                target_pos = start_pos + smooth_ratio * (end_pos - start_pos)
                target_poses.append(target_pos)
            
            return target_poses

    def apply_velocity_limit(self, target_poses):

        """限制关节速度，避免抖动"""

        if self.current_target_poses is None:

            self.current_target_poses = target_poses.copy()

            return target_poses

        

        limited_poses = []

        for i in range(len(target_poses)):

            current_pos = self.current_target_poses[i]

            target_pos = target_poses[i]

            

            # 计算位置差

            pos_diff = target_pos - current_pos

            

            # 限制最大变化量

            max_change = self.max_joint_velocity * self.control_dt_

            if abs(pos_diff) > max_change:

                limited_pos = current_pos + np.sign(pos_diff) * max_change

            else:

                limited_pos = target_pos

            

            limited_poses.append(limited_pos)

        

        self.current_target_poses = limited_poses.copy()

        return limited_poses

    def get_current_joint_positions(self):
        """从low_state获取当前关节位置"""
        if self.low_state is None:
            return [0.0] * len(self.arm_joints)
        return [self.low_state.motor_state[joint].q for joint in self.arm_joints]

    def print_time_info(self):
        """打印时间信息"""
        # 简单显示进度，避免刷屏
        if int(self.time_ * 10) % 10 == 0:  # 每0.1秒打印一次
            print(f"\rTime: {self.time_:.1f}s", end='', flush=True)

    def log_joint_angles(self):
        """记录关节角度（可选功能）"""
        pass

    def check_and_print_action_completion(self):
        """检查动作完成并打印信息"""
        for i in range(len(self.action_sequence)):
            if abs(self.time_ - self.action_sequence[i]["time"]) < 0.05:
                if i != self.last_action_index:
                    print(f"\n[Action {i}] Time: {self.action_sequence[i]['time']}s - Completed")
                    self.last_action_index = i

    def print_joint_angles(self):
        """打印当前关节角度"""
        if self.low_state is None:
            print("Low state not available")
            return
        
        print("\nLeft Arm:")
        print(f"  Shoulder: Pitch={self.low_state.motor_state[G1JointIndex.LeftShoulderPitch].q:.3f}, "
              f"Roll={self.low_state.motor_state[G1JointIndex.LeftShoulderRoll].q:.3f}, "
              f"Yaw={self.low_state.motor_state[G1JointIndex.LeftShoulderYaw].q:.3f}")
        print(f"  Elbow: {self.low_state.motor_state[G1JointIndex.LeftElbow].q:.3f}")
        print(f"  Wrist: Roll={self.low_state.motor_state[G1JointIndex.LeftWristRoll].q:.3f}, "
              f"Pitch={self.low_state.motor_state[G1JointIndex.LeftWristPitch].q:.3f}, "
              f"Yaw={self.low_state.motor_state[G1JointIndex.LeftWristYaw].q:.3f}")
        
        print("\nRight Arm:")
        print(f"  Shoulder: Pitch={self.low_state.motor_state[G1JointIndex.RightShoulderPitch].q:.3f}, "
              f"Roll={self.low_state.motor_state[G1JointIndex.RightShoulderRoll].q:.3f}, "
              f"Yaw={self.low_state.motor_state[G1JointIndex.RightShoulderYaw].q:.3f}")
        print(f"  Elbow: {self.low_state.motor_state[G1JointIndex.RightElbow].q:.3f}")
        print(f"  Wrist: Roll={self.low_state.motor_state[G1JointIndex.RightWristRoll].q:.3f}, "
              f"Pitch={self.low_state.motor_state[G1JointIndex.RightWristPitch].q:.3f}, "
              f"Yaw={self.low_state.motor_state[G1JointIndex.RightWristYaw].q:.3f}")

    def Start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=self.control_dt_, target=self.LowCmdWrite, name="control"
        )
        print("Waiting for robot state...")
        while self.first_update_low_state == False:
            time.sleep(0.1)

        if self.first_update_low_state == True:
            print("Robot state received. Starting control loop...")
            self.lowCmdWriteThreadPtr.Start()

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg

        if self.first_update_low_state == False:
            self.first_update_low_state = True
        
    def LowCmdWrite(self):
        self.time_ += self.control_dt_
        
        # 检查是否收到停止请求
        if self.stop_requested and not self.is_stopping:
            self.is_stopping = True
            self.stop_time = self.time_
            print(f"\n\nStop requested at {self.time_:.2f}s, returning to ready position...")
        
        # 显示时间信息
        self.print_time_info()
        
        # 始终锁定腰部关节
        for waist_joint, lock_pos in self.waist_lock_pos.items():
            self.low_cmd.motor_cmd[waist_joint].tau = 0.
            self.low_cmd.motor_cmd[waist_joint].q = lock_pos
            self.low_cmd.motor_cmd[waist_joint].dq = 0.
            self.low_cmd.motor_cmd[waist_joint].kp = self.kp * 1.5
            self.low_cmd.motor_cmd[waist_joint].kd = self.kd * 1.5

        if self.time_ < 1.0:
            # 启用arm_sdk
            self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1
        
        # 如果正在停止并且已经回到准备位置
        if self.is_stopping and self.stop_time is not None:
            if self.time_ - self.stop_time >= self.cycle_duration * 2:
                # 开始释放控制
                release_duration = 3.0
                elapsed = self.time_ - self.stop_time - self.cycle_duration * 2
                
                if elapsed < release_duration:
                    ratio = elapsed / release_duration
                    self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 1.0 - ratio
                    print(f"\rReleasing control: {ratio*100:.1f}%", end='', flush=True)
                else:
                    # 完全释放
                    self.low_cmd.motor_cmd[G1JointIndex.kNotUsedJoint].q = 0
                    if not self.done:
                        print()
                        print("\n=== Final Joint Angles (Release Arm) ===")
                        self.print_joint_angles()
                    self.done = True
                    
                    self.low_cmd.crc = self.crc.Crc(self.low_cmd)
                    self.arm_sdk_publisher.Write(self.low_cmd)
                    return
        
        # 正常执行动作（准备阶段或循环阶段）
        if not self.done:
            target_poses = self.get_interpolated_pose(self.time_)
            # 应用速度限制
            limited_poses = self.apply_velocity_limit(target_poses)
            
            for i, joint in enumerate(self.arm_joints):
                # 使用更合适的控制参数
                joint_kp = self.kp * 0.9  # 降低刚度，减少抖动
                joint_kd = self.kd * 1.1  # 增加阻尼
                
                self.low_cmd.motor_cmd[joint].tau = 0.
                self.low_cmd.motor_cmd[joint].q = limited_poses[i]
                self.low_cmd.motor_cmd[joint].dq = 0.
                self.low_cmd.motor_cmd[joint].kp = joint_kp
                self.low_cmd.motor_cmd[joint].kd = joint_kd

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.arm_sdk_publisher.Write(self.low_cmd)

def keyboard_listener(custom):
    """监听键盘输入的线程"""
    print("\n=== Control Instructions ===")
    print("Type 'ss' and press Enter to stop the robot and return to ready position")
    print("============================\n")
    
    while not custom.done:
        try:
            user_input = input().strip().lower()
            if user_input == 'ss':
                print("\nStop command received!")
                custom.stop_requested = True
                break
        except:
            break

if __name__ == '__main__':
    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    print("The robot will continuously move between point1 and point2 until you type 'ss'")
    input("Press Enter to continue...")

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom()
    custom.Init()
    custom.Start()
    
    # 启动键盘监听线程
    keyboard_thread = threading.Thread(target=keyboard_listener, args=(custom,), daemon=True)
    keyboard_thread.start()

    while True:        
        time.sleep(0.1)
        if custom.done: 
           print("\nDone!")
           sys.exit(-1)    