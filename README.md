# RIPL Summer 2026: Residual RL Finetuning for ManiSkill PushT-v1

**Author:** Siddhesh Girase  
**Project:** Improving Imitation Learning via LLM-Generated Dense Rewards and Residual Policy Finetuning

## Objective
Enhance a visual diffusion policy baseline on the ManiSkill PushT-v1 manipulation task by:
1. Analyzing failure modes quantitatively
2. Generating dense reward functions using LLM
3. Applying residual RL finetuning with custom wrappers

## Tasks Overview

### Task I: Visual Diffusion Baseline
- Download RGBD demonstration dataset
- Train diffusion policy on PushT-v1 environment
- Evaluate baseline success rate across 5 seeds (1,250 rollouts)
- **Baseline Result:** ~0.60% success rate

### Task II: Failure Mode Analysis
- Record failure videos to identify visual patterns
- Characterize two distinct failure modes:
  - **Mode 1 (Inverted):** T-block spawns >90° from target
  - **Mode 2 (Execution):** Over-correction & positional errors
- Quantify success rates per mode across 1,250 rollouts

### Task III: LLM-Generated Dense Reward
- Use Gemini 2.5 Pro to analyze failure videos
- Generate dense reward function encouraging simultaneous translation + rotation
- Implement strict negative penalties to prevent reward farming
- Include robot end-effector distance to prevent "running away"

### Task IV: Residual RL Finetuning
- **Failure-Biased Sampling:** 70% of resets spawn hard configurations
- **Dense Reward Wrapper:** Replace sparse rewards with LLM-generated function
- **Residual Policy:** PPO learns residual actions on top of diffusion baseline
- **Custom Architecture:** 4-channel RGBD CNN with visual feature extraction
- Train on 500K timesteps, evaluate on failure-biased & nominal distributions

## Project Structure
```
RIPL_SidG/
├── RIPL_Summer_26_Siddhesh_Girase.ipynb  # Main notebook (4 tasks)
├── ManiSkill/                             # Local repository
│   ├── custom_reward.py                   # LLM-generated dense reward
│   ├── custom_wrappers.py                 # Batched GPU wrappers
│   ├── finetune_residual_rgb.py          # Modified PPO trainer
│   └── runs/                              # Training checkpoints & logs
├── README.md                              # This file
└── [Documentation files]
```

## Requirements
- ManiSkill 3.0+ with PHYSX-CUDA backend
- PyTorch + torchvision + torchaudio
- Diffusers, zarr, h5py, tensorboard
- Google Colab with L4 GPU (24GB VRAM recommended)

## Quick Start

### In Google Colab:
1. Mount Google Drive
2. Install dependencies (see Task I setup)
3. Run cells sequentially (Task I through Task IV)
4. Monitor training with TensorBoard (Task I-C, Task IV)

### Key Hyperparameters:
- **Baseline:** 50,000 training iterations, batch size 64
- **Finetuning:** 500K timesteps, 16 parallel envs, 50-step rollouts
- **Residual Alpha:** Starts at 1.0, decays over 15K exploration steps
- **Dense Reward:** `-10.0 * dist - 2.0 * angle - 2.0 * tcp_dist`

## Expected Results

| Distribution | Success Rate |
|---|---|
| Baseline (nominal) | ~0.60% |
| Baseline (failure-biased) | ~0.15-0.25% |
| Finetuned (failure-biased) | ~0.40-0.50% (target) |

## Key Innovations

1. **LLM-Powered Reward:** Gemini analyzes failure videos to design rewards
2. **Failure-Biased Sampling:** Oversamples hard configurations during training
3. **Residual Architecture:** Combines learned baseline with RL refinement
4. **Batched GPU Wrappers:** Transparent proxies for Gymnasium environments
5. **Dense Rewards:** Strictly negative, prevents reward farming and escape

## Citation
If using this work, please cite:
```
Girase, S. (2026). Residual RL Finetuning for Robotic Manipulation. RIPL Summer 2026.
```

## References
- [ManiSkill Documentation](https://maniskill.readthedocs.io)
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)
- [OpenAI PPO](https://arxiv.org/abs/1707.06347)

## Contact
For questions or issues, contact: [sgirase3@gatech.edu]

---
**Last Updated:** May 15, 2026
