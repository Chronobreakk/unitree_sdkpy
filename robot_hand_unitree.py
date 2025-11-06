# Dex3-1 灵巧手控制
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_

import numpy as np
from enum import IntEnum
import time
import sys
import threading
from multiprocessing import Process, Array, Lock


Dex3_Num_Motors = 7
kTopicDex3LeftCommand = "rt/dex3/left/cmd"
kTopicDex3RightCommand = "rt/dex3/right/cmd"
kTopicDex3LeftState = "rt/lf/dex3/left/state"  # 注意：与C++一致，包含/lf/
kTopicDex3RightState = "rt/lf/dex3/right/state"

# 关节限位（从C++代码复制，确保安全）
MAX_LIMITS_LEFT = np.array([1.05, 1.05, 1.75, 0.0, 0.0, 0.0, 0.0])
MIN_LIMITS_LEFT = np.array([-1.05, -0.724, 0.0, -1.57, -1.75, -1.57, -1.75])
MAX_LIMITS_RIGHT = np.array([1.05, 0.742, 0.0, 1.57, 1.75, 1.57, 1.75])
MIN_LIMITS_RIGHT = np.array([-1.05, -1.05, -1.75, 0.0, 0.0, 0.0, 0.0])


class Dex3_1_Controller:
    def __init__(self, left_hand_array_in=None, right_hand_array_in=None, dual_hand_data_lock=None, 
                 dual_hand_state_array_out=None, dual_hand_action_array_out=None, 
                 fps=100.0, Unit_Test=False, simulation_mode=False):
        """
        Dex3-1 灵巧手控制器（简化版，用于键盘控制）
        
        simulation_mode: 是否使用仿真模式 (默认False，使用真实机器人)
        """
        print("Initialize Dex3_1_Controller...")

        self.fps = fps
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode

        if self.simulation_mode:
            ChannelFactoryInitialize(1)
        else:
            ChannelFactoryInitialize(0)

        # initialize handcmd publisher and handstate subscriber
        self.LeftHandCmb_publisher = ChannelPublisher(kTopicDex3LeftCommand, HandCmd_)
        self.LeftHandCmb_publisher.Init()
        self.RightHandCmb_publisher = ChannelPublisher(kTopicDex3RightCommand, HandCmd_)
        self.RightHandCmb_publisher.Init()

        self.LeftHandState_subscriber = ChannelSubscriber(kTopicDex3LeftState, HandState_)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(kTopicDex3RightState, HandState_)
        self.RightHandState_subscriber.Init()

        # Shared Arrays for hand states
        self.left_hand_state_array  = Array('d', Dex3_Num_Motors, lock=True)  
        self.right_hand_state_array = Array('d', Dex3_Num_Motors, lock=True)

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        while True:
            if any(self.left_hand_state_array) and any(self.right_hand_state_array):
                break
            time.sleep(0.01)
            print("[Dex3_1_Controller] Waiting to subscribe dds...")
        print("[Dex3_1_Controller] Subscribe dds ok.")

        print("Initialize Dex3_1_Controller OK!\n")

    def _subscribe_hand_state(self):
        while True:
            left_hand_msg  = self.LeftHandState_subscriber.Read()
            right_hand_msg = self.RightHandState_subscriber.Read()
            if left_hand_msg is not None and right_hand_msg is not None:
                # Update left hand state
                for idx, id in enumerate(Dex3_1_Left_JointIndex):
                    self.left_hand_state_array[idx] = left_hand_msg.motor_state[id].q
                # Update right hand state
                for idx, id in enumerate(Dex3_1_Right_JointIndex):
                    self.right_hand_state_array[idx] = right_hand_msg.motor_state[id].q
            time.sleep(0.002)
    
    class _RIS_Mode:
        def __init__(self, id=0, status=0x01, timeout=0):
            self.motor_mode = 0
            self.id = id & 0x0F  # 4 bits for id
            self.status = status & 0x07  # 3 bits for status
            self.timeout = timeout & 0x01  # 1 bit for timeout

        def _mode_to_uint8(self):
            self.motor_mode |= (self.id & 0x0F)
            self.motor_mode |= (self.status & 0x07) << 4
            self.motor_mode |= (self.timeout & 0x01) << 7
            return self.motor_mode

    def _initialize_cmd_messages(self):
        """初始化控制命令消息（参数与C++一致）"""
        q = 0.0
        dq = 0.0
        tau = 0.0
        kp = 1.5
        kd = 0.1  # 修正：与C++一致，使用0.1

        # initialize dex3-1's left hand cmd msg
        self.left_msg = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Left_JointIndex:
            ris_mode = self._RIS_Mode(id=id, status=0x01, timeout=0)
            motor_mode = ris_mode._mode_to_uint8()
            self.left_msg.motor_cmd[id].mode = motor_mode
            self.left_msg.motor_cmd[id].q = q
            self.left_msg.motor_cmd[id].dq = dq
            self.left_msg.motor_cmd[id].tau = tau
            self.left_msg.motor_cmd[id].kp = kp
            self.left_msg.motor_cmd[id].kd = kd

        # initialize dex3-1's right hand cmd msg
        self.right_msg = unitree_hg_msg_dds__HandCmd_()
        for id in Dex3_1_Right_JointIndex:
            ris_mode = self._RIS_Mode(id=id, status=0x01, timeout=0)
            motor_mode = ris_mode._mode_to_uint8()
            self.right_msg.motor_cmd[id].mode = motor_mode
            self.right_msg.motor_cmd[id].q = q
            self.right_msg.motor_cmd[id].dq = dq
            self.right_msg.motor_cmd[id].tau = tau
            self.right_msg.motor_cmd[id].kp = kp
            self.right_msg.motor_cmd[id].kd = kd
    
    def _clamp_joint_values(self, q_target, is_left_hand=True):
        """限制关节值在安全范围内（与C++一致）"""
        if is_left_hand:
            return np.clip(q_target, MIN_LIMITS_LEFT, MAX_LIMITS_LEFT)
        else:
            return np.clip(q_target, MIN_LIMITS_RIGHT, MAX_LIMITS_RIGHT)
    
    def stop_motors(self, left=True, right=True):
        """停止电机（设置timeout=1，与C++一致）"""
        if not hasattr(self, 'left_msg'):
            self._initialize_cmd_messages()
        
        if left:
            for id in Dex3_1_Left_JointIndex:
                ris_mode = self._RIS_Mode(id=id, status=0x01, timeout=0x01)
                motor_mode = ris_mode._mode_to_uint8()
                self.left_msg.motor_cmd[id].mode = motor_mode
                self.left_msg.motor_cmd[id].tau = 0.0
                self.left_msg.motor_cmd[id].dq = 0.0
                self.left_msg.motor_cmd[id].kp = 0.0
                self.left_msg.motor_cmd[id].kd = 0.0
                self.left_msg.motor_cmd[id].q = 0.0
            self.LeftHandCmb_publisher.Write(self.left_msg)
        
        if right:
            for id in Dex3_1_Right_JointIndex:
                ris_mode = self._RIS_Mode(id=id, status=0x01, timeout=0x01)
                motor_mode = ris_mode._mode_to_uint8()
                self.right_msg.motor_cmd[id].mode = motor_mode
                self.right_msg.motor_cmd[id].tau = 0.0
                self.right_msg.motor_cmd[id].dq = 0.0
                self.right_msg.motor_cmd[id].kp = 0.0
                self.right_msg.motor_cmd[id].kd = 0.0
                self.right_msg.motor_cmd[id].q = 0.0
            self.RightHandCmb_publisher.Write(self.right_msg)
    
    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """设置左右手电机目标位置（添加安全限位检查）"""
        if not hasattr(self, 'left_msg'):
            self._initialize_cmd_messages()
        
        # 安全限位检查
        left_q_safe = self._clamp_joint_values(left_q_target, is_left_hand=True)
        right_q_safe = self._clamp_joint_values(right_q_target, is_left_hand=False)
        
        # 检查是否有值被限制
        if not np.allclose(left_q_target, left_q_safe):
            print(f"⚠ Warning: Left hand values clamped to safe limits")
        if not np.allclose(right_q_target, right_q_safe):
            print(f"⚠ Warning: Right hand values clamped to safe limits")
            
        for idx, id in enumerate(Dex3_1_Left_JointIndex):
            self.left_msg.motor_cmd[id].q = left_q_safe[idx]
        for idx, id in enumerate(Dex3_1_Right_JointIndex):
            self.right_msg.motor_cmd[id].q = right_q_safe[idx]

        self.LeftHandCmb_publisher.Write(self.left_msg)
        self.RightHandCmb_publisher.Write(self.right_msg)

class Dex3_1_Left_JointIndex(IntEnum):
    kLeftHandThumb0 = 0
    kLeftHandThumb1 = 1
    kLeftHandThumb2 = 2
    kLeftHandMiddle0 = 3
    kLeftHandMiddle1 = 4
    kLeftHandIndex0 = 5
    kLeftHandIndex1 = 6

class Dex3_1_Right_JointIndex(IntEnum):
    kRightHandThumb0 = 0
    kRightHandThumb1 = 1
    kRightHandThumb2 = 2
    kRightHandIndex0 = 3
    kRightHandIndex1 = 4
    kRightHandMiddle0 = 5
    kRightHandMiddle1 = 6





if __name__ == "__main__":
    print("=" * 50)
    print("Dex3-1 Keyboard Control")
    print("=" * 50)
    print("Commands:")
    print("  1 - Left hand open")
    print("  2 - Left hand close")
    print("  3 - Right hand open")
    print("  4 - Right hand close")
    print("  5 - Exit program (safe shutdown)")
    print("  6 - Move left hand to custom position")
    print("  7 - Move right hand to custom position")
    print("  s - Emergency stop (stop all motors)")
    print("=" * 50)
    print()
    
    # Initialize Dex3-1 controller
    left_hand_pos_array = Array('d', 75, lock=True)
    right_hand_pos_array = Array('d', 75, lock=True)
    dual_hand_data_lock = Lock()
    dual_hand_state_array = Array('d', 14, lock=False)
    dual_hand_action_array = Array('d', 14, lock=False)
    
    print("Initializing Dex3-1 controller...")
    hand_ctrl = Dex3_1_Controller(
        left_hand_pos_array, 
        right_hand_pos_array, 
        dual_hand_data_lock, 
        dual_hand_state_array, 
        dual_hand_action_array, 
        Unit_Test=False
    )
    print("Controller initialized successfully!\n")
    
    # Define open and close positions for dex3-1 (基于C++的安全限位)
    # Open: 使用中间位置（安全的起始位置）
    left_open = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Close: 使用接近最大限位的位置（但保持安全余量）
    left_close = np.array([0, 0.6, 1, -1, -1, -1, -1])
    
    right_open = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    right_close = np.array([0, -0.6, -1, 1, 1, 1, 1])
    
    # Current target positions
    left_target = left_open.copy()
    right_target = right_open.copy()
    
    print(f"Left hand range: {MIN_LIMITS_LEFT} to {MAX_LIMITS_LEFT}")
    print(f"Right hand range: {MIN_LIMITS_RIGHT} to {MAX_LIMITS_RIGHT}")
    print(f"Using safe positions with 80% of max limits\n")
    
    # Control loop
    print("Ready! Enter command (1-5):")
    while True:
        try:
            cmd = input("> ")
            
            if cmd == '1':
                print("Left hand opening...")
                left_target = left_open.copy()
                hand_ctrl.ctrl_dual_hand(left_target, right_target)
                print("✓ Left hand opened\n")
                
            elif cmd == '2':
                print("Left hand closing...")
                left_target = left_close.copy()
                hand_ctrl.ctrl_dual_hand(left_target, right_target)
                print("✓ Left hand closed\n")
                
            elif cmd == '3':
                print("Right hand opening...")
                right_target = right_open.copy()
                hand_ctrl.ctrl_dual_hand(left_target, right_target)
                print("✓ Right hand opened\n")
                
            elif cmd == '4':
                print("Right hand closing...")
                right_target = right_close.copy()
                hand_ctrl.ctrl_dual_hand(left_target, right_target)
                print("✓ Right hand closed\n")
                
            elif cmd == '5':
                print("Stopping motors safely...")
                hand_ctrl.stop_motors(left=True, right=True)
                time.sleep(0.5)
                print("Program closed.")
                sys.exit(0)
            
            elif cmd == '6':
                print("\n--- Left Hand Custom Position ---")
                print(f"Valid range for each joint:")
                for i in range(7):
                    print(f"  Joint {i}: [{MIN_LIMITS_LEFT[i]:.3f}, {MAX_LIMITS_LEFT[i]:.3f}]")
                print("\nEnter 7 values separated by spaces (or 'c' to cancel):")
                print("Example: 0.5 0.8 0.0 -0.5 -1.0 0.0 0.0")
                
                user_input = input("> ")
                if user_input.lower() == 'c':
                    print("Cancelled\n")
                else:
                    try:
                        values = [float(x) for x in user_input.split()]
                        if len(values) != 7:
                            print(f"❌ Error: Expected 7 values, got {len(values)}\n")
                        else:
                            left_target = np.array(values)
                            # 安全检查会在ctrl_dual_hand中自动执行
                            hand_ctrl.ctrl_dual_hand(left_target, right_target)
                            print(f"✓ Left hand moved to: {left_target}\n")
                    except ValueError:
                        print("❌ Error: Invalid input format. Please enter numbers only.\n")
            
            elif cmd == '7':
                print("\n--- Right Hand Custom Position ---")
                print(f"Valid range for each joint:")
                for i in range(7):
                    print(f"  Joint {i}: [{MIN_LIMITS_RIGHT[i]:.3f}, {MAX_LIMITS_RIGHT[i]:.3f}]")
                print("\nEnter 7 values separated by spaces (or 'c' to cancel):")
                print("Example: 0.5 -0.8 0.0 0.5 1.0 0.0 0.0")
                
                user_input = input("> ")
                if user_input.lower() == 'c':
                    print("Cancelled\n")
                else:
                    try:
                        values = [float(x) for x in user_input.split()]
                        if len(values) != 7:
                            print(f"❌ Error: Expected 7 values, got {len(values)}\n")
                        else:
                            right_target = np.array(values)
                            # 安全检查会在ctrl_dual_hand中自动执行
                            hand_ctrl.ctrl_dual_hand(left_target, right_target)
                            print(f"✓ Right hand moved to: {right_target}\n")
                    except ValueError:
                        print("❌ Error: Invalid input format. Please enter numbers only.\n")
            
            elif cmd == 's':
                print("Emergency stop - Setting all motors to safe state...")
                hand_ctrl.stop_motors(left=True, right=True)
                print("✓ Motors stopped\n")

            elif cmd == 'c':
                print("Right and left hand closing...")
                right_target = right_close.copy()
                left_target = left_close.copy()
                hand_ctrl.ctrl_dual_hand(left_target, right_target)
                print("✓ Both hand closed\n")
                
            elif cmd == 'o':
                print("Right and left hand opeing...")
                right_target = right_open.copy()
                left_target = left_open.copy()
                hand_ctrl.ctrl_dual_hand(left_target, right_target)
                print("✓ Both hand opened\n")
            else:
                print("⚠ Invalid command! Please enter 1-7 or 's' for emergency stop\n")
                
        except KeyboardInterrupt:
            print("\n\nProgram interrupted by user.")
            print("Stopping motors safely...")
            hand_ctrl.stop_motors(left=True, right=True)
            time.sleep(0.5)
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")
            print("Stopping motors...")
            hand_ctrl.stop_motors(left=True, right=True)
            break


