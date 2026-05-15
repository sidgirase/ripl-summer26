import numpy as np

def compute_dense_reward(obj_pose, goal_pose, tcp_pose=None):
    # 1. Positional penalty (Block to Goal)
    obj_pos = obj_pose.p
    goal_pos = goal_pose.p
    distance_penalty = np.linalg.norm(obj_pos - goal_pos)

    # 2. Angular penalty (Z-axis yaw difference)
    obj_yaw = 2 * np.arctan2(obj_pose.q[3], obj_pose.q[0])
    goal_yaw = 2 * np.arctan2(goal_pose.q[3], goal_pose.q[0])
    raw_angle_diff = goal_yaw - obj_yaw
    angle_diff_wrapped = (raw_angle_diff + np.pi) % (2 * np.pi) - np.pi
    angle_penalty = np.abs(angle_diff_wrapped)
    
    # 3. Approach Penalty (Robot to Block) - STOPS RUNNING AWAY
    tcp_penalty = 0.0
    if tcp_pose is not None:
        tcp_penalty = np.linalg.norm(tcp_pose.p - obj_pos)

    # 4. Return a strictly negative cost
    reward = -(distance_penalty * 10.0) - (angle_penalty * 2.0) - (tcp_penalty * 2.0)

    return float(reward)
