import torch
import numpy as np
from custom_reward import compute_dense_reward

class BaseProxyWrapper:
    """A master transparent proxy that bypasses strict Gymnasium inheritance crashes."""
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        # Block dunder methods to prevent recursion, but allow single-underscore
        # attributes like '_env' which ManiSkill's PPO baseline directly accesses.
        if name.startswith('__'):
            raise AttributeError(f"attempted to get missing magic attribute '{name}'")
        return getattr(self.env, name)

    @property
    def _env(self):
        # Explicitly expose _env for ManiSkill's gym_utils value checking
        return getattr(self.env, '_env', self.env)

    @property
    def unwrapped(self):
        return getattr(self.env, 'unwrapped', self.env)

    @property
    def single_observation_space(self): return self.env.single_observation_space
    @property
    def single_action_space(self): return self.env.single_action_space
    @property
    def num_envs(self): return getattr(self.env, 'num_envs', 1)

    def step(self, action): return self.env.step(action)
    def reset(self, **kwargs): return self.env.reset(**kwargs)
    def close(self): return self.env.close()

class UnifiedRGBDWrapper(BaseProxyWrapper):
    """
    Adds a normalized 4-channel 'rgbd' key for PPO.
    Explicitly overrides single_observation_space to prevent KeyError in storage buffers.
    """
    def _process_obs(self, obs):
        if isinstance(obs, dict) and "rgb" in obs and "depth" in obs:
            # Convert to float and normalize [0, 1] to prevent "unsigned char" runtime error
            rgb_f = obs["rgb"].float() / 255.0
            depth_f = obs["depth"].float()
            obs["rgbd"] = torch.cat([rgb_f, depth_f], dim=-1)
        return obs

    @property
    def single_observation_space(self):
        import gymnasium as gym
        orig_space = self.env.single_observation_space
        new_spaces = dict(orig_space.spaces)
        if "rgb" in new_spaces:
            h, w = new_spaces["rgb"].shape[:2]
            new_spaces["rgbd"] = gym.spaces.Box(low=0.0, high=1.0, shape=(h, w, 4), dtype=np.float32)
        return gym.spaces.Dict(new_spaces)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process_obs(obs), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        obs = self._process_obs(obs)
        if "final_observation" in info:
            info["final_observation"] = self._process_obs(info["final_observation"])
        return obs, reward, term, trunc, info

class BatchedFailureBiasedPushTWrapper(BaseProxyWrapper):
    """Oversamples failure modes for natively batched GPU environments."""
    def __init__(self, env, target_failure_ratio=0.70):
        super().__init__(env)
        self.target_failure_ratio = target_failure_ratio

    def reset(self, *, seed=None, options=None):
        max_tries = 100
        for _ in range(max_tries):
            search_seed = seed if seed is not None else np.random.randint(0, 1000000)
            obs, info = self.env.reset(seed=search_seed, options=options)
            try:
                q_obj = self.unwrapped.tee.pose.q.cpu().numpy()
            except AttributeError:
                q_obj = self.unwrapped.obj.pose.q.cpu().numpy()

            w_obj, z_obj = (q_obj[:, 0], q_obj[:, 3]) if len(q_obj.shape) == 2 else (q_obj[0], q_obj[3])
            theta_obj = 2 * np.arctan2(z_obj, w_obj)

            if hasattr(self.unwrapped, 'goal_pose'):
                q_goal = self.unwrapped.goal_pose.q.cpu().numpy()
                w_goal, z_goal = (q_goal[:, 0], q_goal[:, 3]) if len(q_goal.shape) == 2 else (q_goal[0], q_goal[3])
                theta_goal = 2 * np.arctan2(z_goal, w_goal)
            else:
                theta_goal = np.full_like(theta_obj, -np.pi/2)

            theta_diff = (theta_obj - theta_goal + np.pi) % (2 * np.pi) - np.pi
            hard_ratio = np.mean(np.abs(theta_diff) > (np.pi / 2))
            
            if hard_ratio >= self.target_failure_ratio: return obs, info
            if seed is not None: break
        return obs, info

class BatchedLLMDenseRewardWrapper(BaseProxyWrapper):
    """Replaces simulator sparse rewards with LLM-generated dense rewards."""
    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        try:
            q_obj, p_obj = self.unwrapped.tee.pose.q.cpu().numpy(), self.unwrapped.tee.pose.p.cpu().numpy()
        except AttributeError:
            q_obj, p_obj = self.unwrapped.obj.pose.q.cpu().numpy(), self.unwrapped.obj.pose.p.cpu().numpy()

        if hasattr(self.unwrapped, 'goal_pose'):
            q_goal, p_goal = self.unwrapped.goal_pose.q.cpu().numpy(), self.unwrapped.goal_pose.p.cpu().numpy()
        else:
            q_goal = np.zeros_like(q_obj); q_goal[:, 0] = 1.0; p_goal = np.zeros_like(p_obj)
            
        try:
            # Articulated robot fallback
            tcp_p = self.unwrapped.agent.tcp.pose.p.cpu().numpy()
            tcp_q = self.unwrapped.agent.tcp.pose.q.cpu().numpy()
        except AttributeError:
            try:
                # PushT 2D agent fallback
                tcp_p = self.unwrapped.agent.pose.p.cpu().numpy()
                tcp_q = self.unwrapped.agent.pose.q.cpu().numpy()
            except AttributeError:
                tcp_p = np.zeros_like(p_obj)
                tcp_q = np.zeros_like(q_obj)

        new_rewards = torch.zeros_like(reward)
        class DummyPose:
            def __init__(self, p, q): self.p = p; self.q = q

        for i in range(len(p_obj)):
            op = DummyPose(p_obj[i], q_obj[i])
            gp = DummyPose(p_goal[i], q_goal[i])
            tp = DummyPose(tcp_p[i], tcp_q[i])
            new_rewards[i] = float(compute_dense_reward(op, gp, tp))
            
        # Inject massive terminal success bonus (+50) to anchor the PPO agent's value function!
        if "success" in info:
            new_rewards[info["success"]] += 50.0
            
        return obs, new_rewards.to(reward.device), term, trunc, info

class BatchedPolicyDecoratorWrapper(BaseProxyWrapper):
    """Implements the Residual RL architecture utilizing on-policy deterministic scaling."""
    def __init__(self, env, base_agent, alpha=1.0, exploration_horizon=15000, obs_horizon=2):
        super().__init__(env)
        self.base_agent = base_agent
        self.alpha_max = alpha
        self.exploration_horizon = exploration_horizon
        self.obs_horizon = obs_horizon
        self.step_count = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_buffer = []

    def _clone_obs(self, obs):
        """Prevents in-place GPU tensor mutations from freezing the robot."""
        return {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in obs.items()}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        cloned_obs = self._clone_obs(obs)
        self.current_obs = cloned_obs
        self.obs_buffer = [self._clone_obs(obs) for _ in range(self.obs_horizon)]
        return cloned_obs, info

    def step(self, residual_action):
        self.step_count += 1
        
        # ALWAYS RE-PLAN: Fetch fresh action chunk every step based on current history
        with torch.no_grad():
            stacked_obs = {k: torch.stack([buf[k] for buf in self.obs_buffer], dim=1) for k in self.current_obs.keys()}
            action_chunk = self.base_agent.get_action(stacked_obs)
            
        # DISCARD REMAINDER: Only take the first action from the predicted sequence
        base_action = action_chunk[:, 0, :]

        if not isinstance(residual_action, torch.Tensor):
            residual_action = torch.tensor(residual_action, device=self.device)
            
        # PPO requires exact adherence to its distribution. Randomly masking actions destroys the value function.
        # Instead, we scale the entire residual magnitude deterministically over the horizon.
        current_alpha = self.alpha_max * min(1.0, self.step_count / self.exploration_horizon)
        scaled_res = current_alpha * torch.tanh(residual_action)
        
        # Clamp actions to ensure they don't break the physics engine bounds
        final_action = torch.clamp(base_action + scaled_res, -1.0, 1.0)

        obs, reward, terminated, truncated, info = self.env.step(final_action)
        cloned_obs = self._clone_obs(obs)
        self.current_obs = cloned_obs
        self.obs_buffer.pop(0)
        self.obs_buffer.append(cloned_obs)

        # Clear histories independently for resetting environments
        done = terminated | truncated
        if done.any():
            for i in range(self.obs_horizon - 1):
                for k in self.obs_buffer[i].keys():
                    self.obs_buffer[i][k][done] = cloned_obs[k][done]

        return cloned_obs, reward, terminated, truncated, info
