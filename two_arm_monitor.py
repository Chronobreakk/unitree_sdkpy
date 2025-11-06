"""
Minimal monitor that subscribes to `rt/lowstate` and prints left/right 7-joint poses every 1 second.
Removed all arm-motion control code to make this file read-only monitor.
"""

import sys
import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

# Joint indices for left/right 7 joints
LEFT_IDS = [15, 16, 17, 18, 19, 20, 21]
RIGHT_IDS = [22, 23, 24, 25, 26, 27, 28]


def main():
    print("Simple lowstate monitor — prints left/right 7-joint poses every 1s")
    input("Press Enter to continue... (Ctrl+C to abort)")

    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    sub = ChannelSubscriber("rt/lowstate", LowState_)
    sub.Init()

    try:
        while True:
            msg = sub.Read()
            if msg is not None:
                try:
                    left = [msg.motor_state[i].q for i in LEFT_IDS]
                    right = [msg.motor_state[i].q for i in RIGHT_IDS]
                    now = time.strftime('%H:%M:%S')
                    left_str = '[' + ', '.join(f"{v:.3f}" for v in left) + ']'
                    right_str = '[' + ', '.join(f"{v:.3f}" for v in right) + ']'
                    print(f"[{now}] Left: {left_str}    Right: {right_str}")
                except Exception:
                    print("Received lowstate but motor_state is incomplete or malformed")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nMonitor stopped by user")


if __name__ == '__main__':
    main()