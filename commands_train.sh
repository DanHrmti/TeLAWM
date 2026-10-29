# Training commands of all experiments in the paper, one line per experiment family (task x method).
# Not meant to be run as a whole: set the variables and run the lines you need, e.g.
#   SEED=1 IQL_SEED=0; python train.py ...
#
#   ${SEED}      seed of the run, saved to output/train/<agent>_<task>_<SEED>[_<run_suffix>]
#   ${IQL_SEED}  seed of the IQL run of the same task, whose value function and low-level policy the method uses
#   ${AE_SEED}   visual tasks: seed of the autoencoder run of the same task (obs=latent and the high level of HIQL)
#
# Train the IQL of a task first (and, for the visual tasks, the autoencoder); the other methods load it.
# The paper reports 5 seeds per state-based task, each over a different IQL seed, and 3 seeds per visual task.
# Visual SD and Te-LAWM are trained on autoencoder latents and evaluated from pixels by the evaluate.py line below them.

#### PointMaze Medium Navigate ####

## IQL ##
python train.py steps=1001000 task=pointmaze-medium-navigate-v0 gamma=0.99 agent=iql iql.expectile=0.7 iql.alpha=0.01 seed=${SEED}

## TD-MPC ##
python train.py steps=1001000 task=pointmaze-medium-navigate-v0 agent=tdmpc tdmpc.horizon_plan=25 tdmpc.plan_execute_steps=3 tdmpc.pretrained_vf=true tdmpc.vf_chkpt=output/train/iql_pointmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt tdmpc.pretrained_policy=true tdmpc.policy_chkpt=output/train/iql_pointmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SAW ##
python train.py steps=1001000 task=pointmaze-medium-navigate-v0 gamma=0.99 agent=saw saw.alpha=3.0 saw.beta=3.0 saw.subgoal_k=25 saw.pretrained_vf=true saw.vf_chkpt=output/train/iql_pointmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt saw.pretrained_policy=false saw.pi_extract_type="awr" seed=${SEED}

## HIQL ##
python train.py steps=1001000 task=pointmaze-medium-navigate-v0 agent=hiql hiql.beta=3.0 hiql.subgoal_k=25 hiql.subgoal_steps=10 hiql.subgoal_reached_threshold=3 hiql.goal_vicinity_threshold=30 hiql.pretrained_vf=true hiql.vf_chkpt=output/train/iql_pointmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt hiql.pretrained_policy=true hiql.policy_chkpt=output/train/iql_pointmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SD ##
python train.py steps=1001000 task=pointmaze-medium-navigate-v0 agent=sd sd.goal_conditioned=false sd.n_diffusion_steps=20 sd.ddim_sampling_steps=10 sd.sampling_temperature=1.0 sd.subgoal_k=25 sd.n_sg_samples=128 sd.subgoal_steps=10 sd.subgoal_reached_threshold=3 sd.goal_vicinity_threshold=30 sd.pretrained_vf=true sd.vf_chkpt=output/train/iql_pointmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt sd.pretrained_policy=true sd.policy_chkpt=output/train/iql_pointmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM ##
python train.py steps=1001000 task=pointmaze-medium-navigate-v0 agent=telawm telawm.latent_dim=2 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=25 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=3 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_pointmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_pointmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM Fixed Prior ##
python train.py wandb_name_suffix="_fixedprior" steps=1001000 task=pointmaze-medium-navigate-v0 agent=telawm telawm.latent_dim=2 telawm.learned_prior=false telawm.beta_kl=0.004 telawm.lam_timescale=25 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=3 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_pointmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_pointmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt run_suffix=fixedprior seed=${SEED}

## Te-LAWM Single-step ##
python train.py wandb_name_suffix="_singlestep" steps=1001000 task=pointmaze-medium-navigate-v0 agent=telawm telawm.latent_dim=2 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=50 telawm.subgoal_steps=10 telawm.planning_steps=1 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=3 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_pointmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_pointmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt run_suffix=singlestep seed=${SEED}

## ACWM ##
python train.py steps=1001000 task=pointmaze-medium-navigate-v0 agent=acwm acwm.wm_timescale=25 acwm.subgoal_steps=10 acwm.planning_steps=2 acwm.subgoal_state=first acwm.n_sg_samples=128 acwm.subgoal_reached_threshold=3 acwm.goal_vicinity_threshold=30 acwm.pretrained_vf=true acwm.vf_chkpt=output/train/iql_pointmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt acwm.pretrained_policy=true acwm.policy_chkpt=output/train/iql_pointmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## LACWM ##
python train.py steps=1001000 task=pointmaze-medium-navigate-v0 agent=lacwm lacwm.latent_dim=2 lacwm.learned_prior=true lacwm.beta_kl=0.004 lacwm.wm_timescale=25 lacwm.state_conditioned=true lacwm.subgoal_steps=10 lacwm.planning_steps=2 lacwm.subgoal_state=first lacwm.n_sg_samples=128 lacwm.subgoal_reached_threshold=3 lacwm.goal_vicinity_threshold=30 lacwm.pretrained_vf=true lacwm.vf_chkpt=output/train/iql_pointmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt lacwm.pretrained_policy=true lacwm.policy_chkpt=output/train/iql_pointmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

#### PointMaze Medium Stitch ####

## IQL ##
python train.py steps=1001000 task=pointmaze-medium-stitch-v0 gamma=0.99 agent=iql iql.expectile=0.7 iql.alpha=0.01 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 seed=${SEED}

## TD-MPC ##
python train.py steps=1001000 task=pointmaze-medium-stitch-v0 agent=tdmpc tdmpc.horizon_plan=25 tdmpc.plan_execute_steps=3 tdmpc.pretrained_vf=true tdmpc.vf_chkpt=output/train/iql_pointmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt tdmpc.pretrained_policy=true tdmpc.policy_chkpt=output/train/iql_pointmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SAW ##
python train.py steps=1001000 task=pointmaze-medium-stitch-v0 gamma=0.99 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 agent=saw saw.alpha=3.0 saw.beta=3.0 saw.subgoal_k=25 saw.pretrained_vf=true saw.vf_chkpt=output/train/iql_pointmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt saw.pretrained_policy=false saw.pi_extract_type="awr" seed=${SEED}

## HIQL ##
python train.py steps=1001000 task=pointmaze-medium-stitch-v0 agent=hiql hiql.beta=3.0 hiql.subgoal_k=25 hiql.subgoal_steps=10 hiql.subgoal_reached_threshold=3 hiql.goal_vicinity_threshold=30 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 hiql.pretrained_vf=true hiql.vf_chkpt=output/train/iql_pointmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt hiql.pretrained_policy=true hiql.policy_chkpt=output/train/iql_pointmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SD ##
python train.py steps=1001000 task=pointmaze-medium-stitch-v0 agent=sd sd.goal_conditioned=false sd.n_diffusion_steps=20 sd.ddim_sampling_steps=10 sd.sampling_temperature=1.0 sd.subgoal_k=25 sd.n_sg_samples=128 sd.subgoal_steps=10 sd.subgoal_reached_threshold=3 sd.goal_vicinity_threshold=30 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 sd.pretrained_vf=true sd.vf_chkpt=output/train/iql_pointmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt sd.pretrained_policy=true sd.policy_chkpt=output/train/iql_pointmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM ##
python train.py steps=1001000 task=pointmaze-medium-stitch-v0 agent=telawm telawm.latent_dim=2 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=25 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=3 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_pointmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_pointmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM Fixed Prior ##
python train.py wandb_name_suffix="_fixedprior" steps=1001000 task=pointmaze-medium-stitch-v0 agent=telawm telawm.latent_dim=2 telawm.learned_prior=false telawm.beta_kl=0.004 telawm.lam_timescale=25 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=3 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_pointmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_pointmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt run_suffix=fixedprior seed=${SEED}

## Te-LAWM Single-step ##
python train.py wandb_name_suffix="_singlestep" steps=1001000 task=pointmaze-medium-stitch-v0 agent=telawm telawm.latent_dim=2 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=50 telawm.subgoal_steps=10 telawm.planning_steps=1 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=3 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_pointmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_pointmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt run_suffix=singlestep seed=${SEED}

## ACWM ##
python train.py steps=1001000 task=pointmaze-medium-stitch-v0 agent=acwm acwm.wm_timescale=25 acwm.subgoal_steps=10 acwm.planning_steps=2 acwm.subgoal_state=first acwm.n_sg_samples=128 acwm.subgoal_reached_threshold=3 acwm.goal_vicinity_threshold=30 acwm.pretrained_vf=true acwm.vf_chkpt=output/train/iql_pointmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt acwm.pretrained_policy=true acwm.policy_chkpt=output/train/iql_pointmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## LACWM ##
python train.py steps=1001000 task=pointmaze-medium-stitch-v0 agent=lacwm lacwm.latent_dim=2 lacwm.learned_prior=true lacwm.beta_kl=0.004 lacwm.wm_timescale=25 lacwm.state_conditioned=true lacwm.subgoal_steps=10 lacwm.planning_steps=2 lacwm.subgoal_state=first lacwm.n_sg_samples=128 lacwm.subgoal_reached_threshold=3 lacwm.goal_vicinity_threshold=30 lacwm.pretrained_vf=true lacwm.vf_chkpt=output/train/iql_pointmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt lacwm.pretrained_policy=true lacwm.policy_chkpt=output/train/iql_pointmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

#### PointMaze Large Navigate ####

## IQL ##
python train.py steps=1001000 task=pointmaze-large-navigate-v0 gamma=0.999 agent=iql iql.expectile=0.7 iql.alpha=0.01 seed=${SEED}

## TD-MPC ##
python train.py steps=1001000 task=pointmaze-large-navigate-v0 agent=tdmpc tdmpc.horizon_plan=25 tdmpc.plan_execute_steps=3 tdmpc.pretrained_vf=true tdmpc.vf_chkpt=output/train/iql_pointmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt tdmpc.pretrained_policy=true tdmpc.policy_chkpt=output/train/iql_pointmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SAW ##
python train.py steps=1001000 task=pointmaze-large-navigate-v0 gamma=0.999 p_trajgoal_policy=1.0 p_randgoal_policy=0.0 agent=saw saw.alpha=3.0 saw.beta=3.0 saw.subgoal_k=25 saw.pretrained_vf=true saw.vf_chkpt=output/train/iql_pointmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt saw.pretrained_policy=false saw.pi_extract_type="awr" seed=${SEED}

## HIQL ##
python train.py steps=1001000 task=pointmaze-large-navigate-v0 agent=hiql hiql.beta=3.0 hiql.subgoal_k=25 hiql.subgoal_steps=10 hiql.subgoal_reached_threshold=3 hiql.goal_vicinity_threshold=30 hiql.pretrained_vf=true hiql.vf_chkpt=output/train/iql_pointmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt hiql.pretrained_policy=true hiql.policy_chkpt=output/train/iql_pointmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SD ##
python train.py steps=1001000 task=pointmaze-large-navigate-v0 agent=sd sd.goal_conditioned=false sd.n_diffusion_steps=20 sd.ddim_sampling_steps=10 sd.sampling_temperature=1.0 sd.subgoal_k=100 sd.n_sg_samples=128 sd.subgoal_steps=10 sd.subgoal_reached_threshold=3 sd.goal_vicinity_threshold=30 sd.pretrained_vf=true sd.vf_chkpt=output/train/iql_pointmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt sd.pretrained_policy=true sd.policy_chkpt=output/train/iql_pointmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM ##
python train.py steps=1001000 task=pointmaze-large-navigate-v0 agent=telawm telawm.latent_dim=2 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=100 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=3 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_pointmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_pointmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM Fixed Prior ##
python train.py wandb_name_suffix="_fixedprior" steps=1001000 task=pointmaze-large-navigate-v0 agent=telawm telawm.latent_dim=2 telawm.learned_prior=false telawm.beta_kl=0.004 telawm.lam_timescale=100 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=3 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_pointmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_pointmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt run_suffix=fixedprior seed=${SEED}

## Te-LAWM Single-step ##
python train.py wandb_name_suffix="_singlestep" steps=1001000 task=pointmaze-large-navigate-v0 agent=telawm telawm.latent_dim=2 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=200 telawm.subgoal_steps=10 telawm.planning_steps=1 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=3 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_pointmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_pointmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt run_suffix=singlestep seed=${SEED}

## ACWM ##
python train.py steps=1001000 task=pointmaze-large-navigate-v0 agent=acwm acwm.wm_timescale=100 acwm.subgoal_steps=10 acwm.planning_steps=2 acwm.subgoal_state=first acwm.n_sg_samples=128 acwm.subgoal_reached_threshold=3 acwm.goal_vicinity_threshold=30 acwm.pretrained_vf=true acwm.vf_chkpt=output/train/iql_pointmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt acwm.pretrained_policy=true acwm.policy_chkpt=output/train/iql_pointmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## LACWM ##
python train.py steps=1001000 task=pointmaze-large-navigate-v0 agent=lacwm lacwm.latent_dim=2 lacwm.learned_prior=true lacwm.beta_kl=0.004 lacwm.wm_timescale=100 lacwm.state_conditioned=true lacwm.subgoal_steps=10 lacwm.planning_steps=2 lacwm.subgoal_state=first lacwm.n_sg_samples=128 lacwm.subgoal_reached_threshold=3 lacwm.goal_vicinity_threshold=30 lacwm.pretrained_vf=true lacwm.vf_chkpt=output/train/iql_pointmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt lacwm.pretrained_policy=true lacwm.policy_chkpt=output/train/iql_pointmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

#### PointMaze Large Stitch ####

## IQL ##
python train.py steps=1001000 task=pointmaze-large-stitch-v0 gamma=0.999 agent=iql iql.expectile=0.7 iql.alpha=0.01 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 seed=${SEED}

## TD-MPC ##
python train.py steps=1001000 task=pointmaze-large-stitch-v0 agent=tdmpc tdmpc.horizon_plan=25 tdmpc.plan_execute_steps=3 tdmpc.pretrained_vf=true tdmpc.vf_chkpt=output/train/iql_pointmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt tdmpc.pretrained_policy=true tdmpc.policy_chkpt=output/train/iql_pointmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SAW ##
python train.py steps=1001000 task=pointmaze-large-stitch-v0 gamma=0.999 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 agent=saw saw.alpha=3.0 saw.beta=3.0 saw.subgoal_k=25 saw.pretrained_vf=true saw.vf_chkpt=output/train/iql_pointmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt saw.pretrained_policy=false saw.pi_extract_type="awr" seed=${SEED}

## HIQL ##
python train.py steps=1001000 task=pointmaze-large-stitch-v0 agent=hiql hiql.beta=3.0 hiql.subgoal_k=25 hiql.subgoal_steps=10 hiql.subgoal_reached_threshold=3 hiql.goal_vicinity_threshold=30 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 hiql.pretrained_vf=true hiql.vf_chkpt=output/train/iql_pointmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt hiql.pretrained_policy=true hiql.policy_chkpt=output/train/iql_pointmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SD ##
python train.py steps=1001000 task=pointmaze-large-stitch-v0 agent=sd sd.goal_conditioned=false sd.n_diffusion_steps=20 sd.ddim_sampling_steps=10 sd.sampling_temperature=1.0 sd.subgoal_k=100 sd.n_sg_samples=128 sd.subgoal_steps=10 sd.subgoal_reached_threshold=3 sd.goal_vicinity_threshold=30 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 sd.pretrained_vf=true sd.vf_chkpt=output/train/iql_pointmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt sd.pretrained_policy=true sd.policy_chkpt=output/train/iql_pointmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM ##
python train.py steps=1001000 task=pointmaze-large-stitch-v0 agent=telawm telawm.latent_dim=2 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=100 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=3 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_pointmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_pointmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM Fixed Prior ##
python train.py wandb_name_suffix="_fixedprior" steps=1001000 task=pointmaze-large-stitch-v0 agent=telawm telawm.latent_dim=2 telawm.learned_prior=false telawm.beta_kl=0.004 telawm.lam_timescale=100 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=3 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_pointmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_pointmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt run_suffix=fixedprior seed=${SEED}

## Te-LAWM Single-step ##
python train.py wandb_name_suffix="_singlestep" steps=1001000 task=pointmaze-large-stitch-v0 agent=telawm telawm.latent_dim=2 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=100 telawm.subgoal_steps=10 telawm.planning_steps=1 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=3 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_pointmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_pointmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt run_suffix=singlestep seed=${SEED}

## ACWM ##
python train.py steps=1001000 task=pointmaze-large-stitch-v0 agent=acwm acwm.wm_timescale=100 acwm.subgoal_steps=10 acwm.planning_steps=2 acwm.subgoal_state=first acwm.n_sg_samples=128 acwm.subgoal_reached_threshold=3 acwm.goal_vicinity_threshold=30 acwm.pretrained_vf=true acwm.vf_chkpt=output/train/iql_pointmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt acwm.pretrained_policy=true acwm.policy_chkpt=output/train/iql_pointmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## LACWM ##
python train.py steps=1001000 task=pointmaze-large-stitch-v0 agent=lacwm lacwm.latent_dim=2 lacwm.learned_prior=true lacwm.beta_kl=0.004 lacwm.wm_timescale=100 lacwm.state_conditioned=true lacwm.subgoal_steps=10 lacwm.planning_steps=2 lacwm.subgoal_state=first lacwm.n_sg_samples=128 lacwm.subgoal_reached_threshold=3 lacwm.goal_vicinity_threshold=30 lacwm.pretrained_vf=true lacwm.vf_chkpt=output/train/iql_pointmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt lacwm.pretrained_policy=true lacwm.policy_chkpt=output/train/iql_pointmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

#### AntMaze Medium Navigate ####

## IQL ##
python train.py steps=1001000 task=antmaze-medium-navigate-v0 gamma=0.99 agent=iql iql.expectile=0.7 iql.alpha=0.1 seed=${SEED}

## TD-MPC ##
python train.py steps=1001000 task=antmaze-medium-navigate-v0 agent=tdmpc tdmpc.horizon_plan=25 tdmpc.plan_execute_steps=3 tdmpc.pretrained_vf=true tdmpc.vf_chkpt=output/train/iql_antmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt tdmpc.pretrained_policy=true tdmpc.policy_chkpt=output/train/iql_antmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SAW ##
python train.py steps=1001000 task=antmaze-medium-navigate-v0 gamma=0.99 p_trajgoal_policy=1.0 p_randgoal_policy=0.0 agent=saw saw.alpha=3.0 saw.beta=3.0 saw.subgoal_k=25 saw.pretrained_vf=true saw.vf_chkpt=output/train/iql_antmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt saw.pretrained_policy=false saw.pi_extract_type="awr" seed=${SEED}

## HIQL ##
python train.py steps=1001000 task=antmaze-medium-navigate-v0 agent=hiql hiql.beta=3.0 hiql.subgoal_k=25 hiql.subgoal_steps=10 hiql.subgoal_reached_threshold=10 hiql.goal_vicinity_threshold=30 hiql.pretrained_vf=true hiql.vf_chkpt=output/train/iql_antmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt hiql.pretrained_policy=true hiql.policy_chkpt=output/train/iql_antmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SD ##
python train.py steps=1001000 task=antmaze-medium-navigate-v0 agent=sd sd.goal_conditioned=false sd.n_diffusion_steps=20 sd.ddim_sampling_steps=10 sd.sampling_temperature=1.0 sd.subgoal_k=25 sd.n_sg_samples=128 sd.subgoal_steps=10 sd.subgoal_reached_threshold=10 sd.goal_vicinity_threshold=30 sd.pretrained_vf=true sd.vf_chkpt=output/train/iql_antmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt sd.pretrained_policy=true sd.policy_chkpt=output/train/iql_antmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM ##
python train.py steps=1001000 task=antmaze-medium-navigate-v0 agent=telawm telawm.latent_dim=8 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=25 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_antmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_antmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM Fixed Prior ##
python train.py wandb_name_suffix="_fixedprior" steps=1001000 task=antmaze-medium-navigate-v0 agent=telawm telawm.latent_dim=8 telawm.learned_prior=false telawm.beta_kl=0.004 telawm.lam_timescale=25 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_antmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_antmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt run_suffix=fixedprior seed=${SEED}

## Te-LAWM Single-step ##
python train.py wandb_name_suffix="_singlestep" steps=1001000 task=antmaze-medium-navigate-v0 agent=telawm telawm.latent_dim=8 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=50 telawm.subgoal_steps=10 telawm.planning_steps=1 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_antmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_antmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt run_suffix=singlestep seed=${SEED}

## ACWM ##
python train.py steps=1001000 task=antmaze-medium-navigate-v0 agent=acwm acwm.wm_timescale=25 acwm.subgoal_steps=10 acwm.planning_steps=2 acwm.subgoal_state=first acwm.n_sg_samples=128 acwm.subgoal_reached_threshold=10 acwm.goal_vicinity_threshold=30 acwm.pretrained_vf=true acwm.vf_chkpt=output/train/iql_antmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt acwm.pretrained_policy=true acwm.policy_chkpt=output/train/iql_antmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## LACWM ##
python train.py steps=1001000 task=antmaze-medium-navigate-v0 agent=lacwm lacwm.latent_dim=8 lacwm.learned_prior=true lacwm.beta_kl=0.004 lacwm.wm_timescale=25 lacwm.state_conditioned=true lacwm.subgoal_steps=10 lacwm.planning_steps=2 lacwm.subgoal_state=first lacwm.n_sg_samples=128 lacwm.subgoal_reached_threshold=10 lacwm.goal_vicinity_threshold=30 lacwm.pretrained_vf=true lacwm.vf_chkpt=output/train/iql_antmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt lacwm.pretrained_policy=true lacwm.policy_chkpt=output/train/iql_antmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

#### AntMaze Medium Stitch ####

## IQL ##
python train.py steps=1001000 task=antmaze-medium-stitch-v0 gamma=0.99 agent=iql iql.expectile=0.7 iql.alpha=0.1 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 seed=${SEED}

## TD-MPC ##
python train.py steps=1001000 task=antmaze-medium-stitch-v0 agent=tdmpc tdmpc.horizon_plan=25 tdmpc.plan_execute_steps=3 tdmpc.pretrained_vf=true tdmpc.vf_chkpt=output/train/iql_antmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt tdmpc.pretrained_policy=true tdmpc.policy_chkpt=output/train/iql_antmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SAW ##
python train.py steps=1001000 task=antmaze-medium-stitch-v0 gamma=0.99 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 agent=saw saw.alpha=3.0 saw.beta=3.0 saw.subgoal_k=25 saw.pretrained_vf=true saw.vf_chkpt=output/train/iql_antmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt saw.pretrained_policy=false saw.pi_extract_type="awr" seed=${SEED}

## HIQL ##
python train.py steps=1001000 task=antmaze-medium-stitch-v0 agent=hiql hiql.beta=3.0 hiql.subgoal_k=25 hiql.subgoal_steps=10 hiql.subgoal_reached_threshold=10 hiql.goal_vicinity_threshold=30 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 hiql.pretrained_vf=true hiql.vf_chkpt=output/train/iql_antmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt hiql.pretrained_policy=true hiql.policy_chkpt=output/train/iql_antmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SD ##
python train.py steps=1001000 task=antmaze-medium-stitch-v0 agent=sd sd.goal_conditioned=false sd.n_diffusion_steps=20 sd.ddim_sampling_steps=10 sd.sampling_temperature=1.0 sd.subgoal_k=25 sd.n_sg_samples=128 sd.subgoal_steps=10 sd.subgoal_reached_threshold=10 sd.goal_vicinity_threshold=30 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 sd.pretrained_vf=true sd.vf_chkpt=output/train/iql_antmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt sd.pretrained_policy=true sd.policy_chkpt=output/train/iql_antmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM ##
python train.py steps=1001000 task=antmaze-medium-stitch-v0 agent=telawm telawm.latent_dim=8 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=25 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_antmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_antmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM Fixed Prior ##
python train.py wandb_name_suffix="_fixedprior" steps=1001000 task=antmaze-medium-stitch-v0 agent=telawm telawm.latent_dim=8 telawm.learned_prior=false telawm.beta_kl=0.004 telawm.lam_timescale=25 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_antmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_antmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt run_suffix=fixedprior seed=${SEED}

## Te-LAWM Single-step ##
python train.py wandb_name_suffix="_singlestep" steps=1001000 task=antmaze-medium-stitch-v0 agent=telawm telawm.latent_dim=8 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=50 telawm.subgoal_steps=10 telawm.planning_steps=1 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_antmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_antmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt run_suffix=singlestep seed=${SEED}

## ACWM ##
python train.py steps=1001000 task=antmaze-medium-stitch-v0 agent=acwm acwm.wm_timescale=25 acwm.subgoal_steps=10 acwm.planning_steps=2 acwm.subgoal_state=first acwm.n_sg_samples=128 acwm.subgoal_reached_threshold=10 acwm.goal_vicinity_threshold=30 acwm.pretrained_vf=true acwm.vf_chkpt=output/train/iql_antmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt acwm.pretrained_policy=true acwm.policy_chkpt=output/train/iql_antmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## LACWM ##
python train.py steps=1001000 task=antmaze-medium-stitch-v0 agent=lacwm lacwm.latent_dim=8 lacwm.learned_prior=true lacwm.beta_kl=0.004 lacwm.wm_timescale=25 lacwm.state_conditioned=true lacwm.subgoal_steps=10 lacwm.planning_steps=2 lacwm.subgoal_state=first lacwm.n_sg_samples=128 lacwm.subgoal_reached_threshold=10 lacwm.goal_vicinity_threshold=30 lacwm.pretrained_vf=true lacwm.vf_chkpt=output/train/iql_antmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt lacwm.pretrained_policy=true lacwm.policy_chkpt=output/train/iql_antmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

#### AntMaze Large Navigate ####

## IQL ##
python train.py steps=1001000 task=antmaze-large-navigate-v0 gamma=0.999 agent=iql iql.expectile=0.7 iql.alpha=0.1 seed=${SEED}

## TD-MPC ##
python train.py steps=1001000 task=antmaze-large-navigate-v0 agent=tdmpc tdmpc.horizon_plan=25 tdmpc.plan_execute_steps=3 tdmpc.pretrained_vf=true tdmpc.vf_chkpt=output/train/iql_antmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt tdmpc.pretrained_policy=true tdmpc.policy_chkpt=output/train/iql_antmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SAW ##
python train.py steps=1001000 task=antmaze-large-navigate-v0 gamma=0.999 p_trajgoal_policy=1.0 p_randgoal_policy=0.0 agent=saw saw.alpha=3.0 saw.beta=3.0 saw.subgoal_k=25 saw.pretrained_vf=true saw.vf_chkpt=output/train/iql_antmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt saw.pretrained_policy=false saw.pi_extract_type="awr" seed=${SEED}

## HIQL ##
python train.py steps=1001000 task=antmaze-large-navigate-v0 agent=hiql hiql.beta=3.0 hiql.subgoal_k=25 hiql.subgoal_steps=10 hiql.subgoal_reached_threshold=10 hiql.goal_vicinity_threshold=30 hiql.pretrained_vf=true hiql.vf_chkpt=output/train/iql_antmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt hiql.pretrained_policy=true hiql.policy_chkpt=output/train/iql_antmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SD ##
python train.py steps=1001000 task=antmaze-large-navigate-v0 agent=sd sd.goal_conditioned=false sd.n_diffusion_steps=20 sd.ddim_sampling_steps=10 sd.sampling_temperature=1.0 sd.subgoal_k=50 sd.n_sg_samples=128 sd.subgoal_steps=10 sd.subgoal_reached_threshold=10 sd.goal_vicinity_threshold=30 sd.pretrained_vf=true sd.vf_chkpt=output/train/iql_antmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt sd.pretrained_policy=true sd.policy_chkpt=output/train/iql_antmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM ##
python train.py steps=1001000 task=antmaze-large-navigate-v0 agent=telawm telawm.latent_dim=8 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=50 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_antmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_antmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM Fixed Prior ##
python train.py wandb_name_suffix="_fixedprior" steps=1001000 task=antmaze-large-navigate-v0 agent=telawm telawm.latent_dim=8 telawm.learned_prior=false telawm.beta_kl=0.004 telawm.lam_timescale=25 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_antmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_antmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt run_suffix=fixedprior seed=${SEED}

## Te-LAWM Single-step ##
python train.py wandb_name_suffix="_singlestep" steps=1001000 task=antmaze-large-navigate-v0 agent=telawm telawm.latent_dim=8 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=50 telawm.subgoal_steps=10 telawm.planning_steps=1 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_antmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_antmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt run_suffix=singlestep seed=${SEED}

## ACWM ##
python train.py steps=1001000 task=antmaze-large-navigate-v0 agent=acwm acwm.wm_timescale=25 acwm.subgoal_steps=10 acwm.planning_steps=2 acwm.subgoal_state=first acwm.n_sg_samples=128 acwm.subgoal_reached_threshold=10 acwm.goal_vicinity_threshold=30 acwm.pretrained_vf=true acwm.vf_chkpt=output/train/iql_antmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt acwm.pretrained_policy=true acwm.policy_chkpt=output/train/iql_antmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## LACWM ##
python train.py steps=1001000 task=antmaze-large-navigate-v0 agent=lacwm lacwm.latent_dim=8 lacwm.learned_prior=true lacwm.beta_kl=0.004 lacwm.wm_timescale=25 lacwm.state_conditioned=true lacwm.subgoal_steps=10 lacwm.planning_steps=2 lacwm.subgoal_state=first lacwm.n_sg_samples=128 lacwm.subgoal_reached_threshold=10 lacwm.goal_vicinity_threshold=30 lacwm.pretrained_vf=true lacwm.vf_chkpt=output/train/iql_antmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt lacwm.pretrained_policy=true lacwm.policy_chkpt=output/train/iql_antmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

#### AntMaze Large Stitch ####

## IQL ##
python train.py steps=1001000 task=antmaze-large-stitch-v0 gamma=0.999 agent=iql iql.expectile=0.7 iql.alpha=0.1 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 seed=${SEED}

## TD-MPC ##
python train.py steps=1001000 task=antmaze-large-stitch-v0 agent=tdmpc tdmpc.horizon_plan=25 tdmpc.plan_execute_steps=3 tdmpc.pretrained_vf=true tdmpc.vf_chkpt=output/train/iql_antmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt tdmpc.pretrained_policy=true tdmpc.policy_chkpt=output/train/iql_antmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SAW ##
python train.py steps=1001000 task=antmaze-large-stitch-v0 gamma=0.999 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 agent=saw saw.alpha=3.0 saw.beta=3.0 saw.subgoal_k=25 saw.pretrained_vf=true saw.vf_chkpt=output/train/iql_antmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt saw.pretrained_policy=false saw.pi_extract_type="awr" seed=${SEED}

## HIQL ##
python train.py steps=1001000 task=antmaze-large-stitch-v0 agent=hiql hiql.beta=3.0 hiql.subgoal_k=25 hiql.subgoal_steps=10 hiql.subgoal_reached_threshold=10 hiql.goal_vicinity_threshold=30 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 hiql.pretrained_vf=true hiql.vf_chkpt=output/train/iql_antmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt hiql.pretrained_policy=true hiql.policy_chkpt=output/train/iql_antmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SD ##
python train.py steps=1001000 task=antmaze-large-stitch-v0 agent=sd sd.goal_conditioned=false sd.n_diffusion_steps=20 sd.ddim_sampling_steps=10 sd.sampling_temperature=1.0 sd.subgoal_k=50 sd.n_sg_samples=128 sd.subgoal_steps=10 sd.subgoal_reached_threshold=10 sd.goal_vicinity_threshold=30 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 sd.pretrained_vf=true sd.vf_chkpt=output/train/iql_antmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt sd.pretrained_policy=true sd.policy_chkpt=output/train/iql_antmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM ##
python train.py steps=1001000 task=antmaze-large-stitch-v0 agent=telawm telawm.latent_dim=8 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=50 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_antmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_antmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM Fixed Prior ##
python train.py wandb_name_suffix="_fixedprior" steps=1001000 task=antmaze-large-stitch-v0 agent=telawm telawm.latent_dim=8 telawm.learned_prior=false telawm.beta_kl=0.004 telawm.lam_timescale=50 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_antmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_antmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt run_suffix=fixedprior seed=${SEED}

## Te-LAWM Single-step ##
python train.py wandb_name_suffix="_singlestep" steps=1001000 task=antmaze-large-stitch-v0 agent=telawm telawm.latent_dim=8 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=100 telawm.subgoal_steps=10 telawm.planning_steps=1 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_antmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_antmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt run_suffix=singlestep seed=${SEED}

## ACWM ##
python train.py steps=1001000 task=antmaze-large-stitch-v0 agent=acwm acwm.wm_timescale=50 acwm.subgoal_steps=10 acwm.planning_steps=2 acwm.subgoal_state=first acwm.n_sg_samples=128 acwm.subgoal_reached_threshold=10 acwm.goal_vicinity_threshold=30 acwm.pretrained_vf=true acwm.vf_chkpt=output/train/iql_antmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt acwm.pretrained_policy=true acwm.policy_chkpt=output/train/iql_antmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## LACWM ##
python train.py steps=1001000 task=antmaze-large-stitch-v0 agent=lacwm lacwm.latent_dim=8 lacwm.learned_prior=true lacwm.beta_kl=0.004 lacwm.wm_timescale=50 lacwm.state_conditioned=true lacwm.subgoal_steps=10 lacwm.planning_steps=2 lacwm.subgoal_state=first lacwm.n_sg_samples=128 lacwm.subgoal_reached_threshold=10 lacwm.goal_vicinity_threshold=30 lacwm.pretrained_vf=true lacwm.vf_chkpt=output/train/iql_antmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt lacwm.pretrained_policy=true lacwm.policy_chkpt=output/train/iql_antmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

#### HumanoidMaze Medium Navigate ####

## IQL ##
python train.py steps=1001000 task=humanoidmaze-medium-navigate-v0 gamma=0.999 agent=iql iql.expectile=0.7 iql.alpha=0.4 seed=${SEED}

## TD-MPC ##
python train.py steps=1001000 task=humanoidmaze-medium-navigate-v0 agent=tdmpc tdmpc.horizon_plan=25 tdmpc.plan_execute_steps=3 tdmpc.pretrained_vf=true tdmpc.vf_chkpt=output/train/iql_humanoidmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt tdmpc.pretrained_policy=true tdmpc.policy_chkpt=output/train/iql_humanoidmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SAW ##
python train.py steps=1001000 task=humanoidmaze-medium-navigate-v0 gamma=0.999 p_trajgoal_policy=1.0 p_randgoal_policy=0.0 agent=saw saw.alpha=3.0 saw.beta=3.0 saw.subgoal_k=100 saw.pretrained_vf=true saw.vf_chkpt=output/train/iql_humanoidmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt saw.pretrained_policy=false saw.pi_extract_type="awr" seed=${SEED}

## HIQL ##
python train.py steps=1001000 task=humanoidmaze-medium-navigate-v0 agent=hiql hiql.beta=3.0 hiql.subgoal_k=100 hiql.subgoal_steps=10 hiql.subgoal_reached_threshold=10 hiql.goal_vicinity_threshold=100 hiql.pretrained_vf=true hiql.vf_chkpt=output/train/iql_humanoidmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt hiql.pretrained_policy=true hiql.policy_chkpt=output/train/iql_humanoidmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SD ##
python train.py steps=1001000 task=humanoidmaze-medium-navigate-v0 agent=sd sd.goal_conditioned=false sd.n_diffusion_steps=20 sd.ddim_sampling_steps=10 sd.sampling_temperature=1.0 sd.subgoal_k=100 sd.n_sg_samples=128 sd.subgoal_steps=10 sd.subgoal_reached_threshold=10 sd.goal_vicinity_threshold=100 sd.pretrained_vf=true sd.vf_chkpt=output/train/iql_humanoidmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt sd.pretrained_policy=true sd.policy_chkpt=output/train/iql_humanoidmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM ##
python train.py steps=1001000 task=humanoidmaze-medium-navigate-v0 agent=telawm telawm.latent_dim=21 telawm.learned_prior=true telawm.beta_kl=0.04 telawm.lam_timescale=100 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.subgoal_state=final telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=100 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_humanoidmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_humanoidmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM Fixed Prior ##
python train.py wandb_name_suffix="_fixedprior" steps=1001000 task=humanoidmaze-medium-navigate-v0 agent=telawm telawm.latent_dim=21 telawm.learned_prior=false telawm.beta_kl=0.04 telawm.lam_timescale=100 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=100 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_humanoidmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_humanoidmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt run_suffix=fixedprior seed=${SEED}

## Te-LAWM Single-step ##
python train.py wandb_name_suffix="_singlestep" steps=1001000 task=humanoidmaze-medium-navigate-v0 agent=telawm telawm.latent_dim=21 telawm.learned_prior=true telawm.beta_kl=0.04 telawm.lam_timescale=200 telawm.subgoal_steps=10 telawm.planning_steps=1 telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=100 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_humanoidmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_humanoidmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt run_suffix=singlestep seed=${SEED}

## ACWM ##
python train.py steps=1001000 task=humanoidmaze-medium-navigate-v0 agent=acwm acwm.wm_timescale=100 acwm.subgoal_steps=10 acwm.planning_steps=2 acwm.n_sg_samples=128 acwm.subgoal_reached_threshold=10 acwm.goal_vicinity_threshold=100 acwm.pretrained_vf=true acwm.vf_chkpt=output/train/iql_humanoidmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt acwm.pretrained_policy=true acwm.policy_chkpt=output/train/iql_humanoidmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## LACWM ##
python train.py steps=1001000 task=humanoidmaze-medium-navigate-v0 agent=lacwm lacwm.latent_dim=21 lacwm.learned_prior=true lacwm.beta_kl=0.04 lacwm.wm_timescale=100 lacwm.state_conditioned=true lacwm.subgoal_steps=10 lacwm.planning_steps=2 lacwm.n_sg_samples=128 lacwm.subgoal_reached_threshold=10 lacwm.goal_vicinity_threshold=100 lacwm.pretrained_vf=true lacwm.vf_chkpt=output/train/iql_humanoidmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt lacwm.pretrained_policy=true lacwm.policy_chkpt=output/train/iql_humanoidmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

#### HumanoidMaze Medium Stitch ####

## IQL ##
python train.py steps=1001000 task=humanoidmaze-medium-stitch-v0 gamma=0.999 agent=iql iql.expectile=0.7 iql.alpha=0.4 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 seed=${SEED}

## TD-MPC ##
python train.py steps=1001000 task=humanoidmaze-medium-stitch-v0 agent=tdmpc tdmpc.horizon_plan=25 tdmpc.plan_execute_steps=3 tdmpc.pretrained_vf=true tdmpc.vf_chkpt=output/train/iql_humanoidmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt tdmpc.pretrained_policy=true tdmpc.policy_chkpt=output/train/iql_humanoidmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SAW ##
python train.py steps=1001000 task=humanoidmaze-medium-stitch-v0 gamma=0.999 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 agent=saw saw.alpha=3.0 saw.beta=3.0 saw.subgoal_k=100 saw.pretrained_vf=true saw.vf_chkpt=output/train/iql_humanoidmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt saw.pretrained_policy=false saw.pi_extract_type="awr" seed=${SEED}

## HIQL ##
python train.py steps=1001000 task=humanoidmaze-medium-stitch-v0 agent=hiql hiql.beta=3.0 hiql.subgoal_k=100 hiql.subgoal_steps=10 hiql.subgoal_reached_threshold=10 hiql.goal_vicinity_threshold=100 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 hiql.pretrained_vf=true hiql.vf_chkpt=output/train/iql_humanoidmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt hiql.pretrained_policy=true hiql.policy_chkpt=output/train/iql_humanoidmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SD ##
python train.py steps=1001000 task=humanoidmaze-medium-stitch-v0 agent=sd sd.goal_conditioned=false sd.n_diffusion_steps=20 sd.ddim_sampling_steps=10 sd.sampling_temperature=1.0 sd.subgoal_k=100 sd.n_sg_samples=128 sd.subgoal_steps=10 sd.subgoal_reached_threshold=10 sd.goal_vicinity_threshold=100 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 sd.pretrained_vf=true sd.vf_chkpt=output/train/iql_humanoidmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt sd.pretrained_policy=true sd.policy_chkpt=output/train/iql_humanoidmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM ##
python train.py steps=1001000 task=humanoidmaze-medium-stitch-v0 agent=telawm telawm.latent_dim=21 telawm.learned_prior=true telawm.beta_kl=0.04 telawm.lam_timescale=100 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.subgoal_state=final telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=100 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_humanoidmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_humanoidmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM Fixed Prior ##
python train.py wandb_name_suffix="_fixedprior" steps=1001000 task=humanoidmaze-medium-stitch-v0 agent=telawm telawm.latent_dim=21 telawm.learned_prior=false telawm.beta_kl=0.04 telawm.lam_timescale=100 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=100 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_humanoidmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_humanoidmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt run_suffix=fixedprior seed=${SEED}

## Te-LAWM Single-step ##
python train.py wandb_name_suffix="_singlestep" steps=1001000 task=humanoidmaze-medium-stitch-v0 agent=telawm telawm.latent_dim=21 telawm.learned_prior=true telawm.beta_kl=0.04 telawm.lam_timescale=200 telawm.subgoal_steps=10 telawm.planning_steps=1 telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=100 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_humanoidmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_humanoidmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt run_suffix=singlestep seed=${SEED}

## ACWM ##
python train.py steps=1001000 task=humanoidmaze-medium-stitch-v0 agent=acwm acwm.wm_timescale=100 acwm.subgoal_steps=10 acwm.planning_steps=2 acwm.n_sg_samples=128 acwm.subgoal_reached_threshold=10 acwm.goal_vicinity_threshold=100 acwm.pretrained_vf=true acwm.vf_chkpt=output/train/iql_humanoidmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt acwm.pretrained_policy=true acwm.policy_chkpt=output/train/iql_humanoidmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## LACWM ##
python train.py steps=1001000 task=humanoidmaze-medium-stitch-v0 agent=lacwm lacwm.latent_dim=21 lacwm.learned_prior=true lacwm.beta_kl=0.04 lacwm.wm_timescale=100 lacwm.state_conditioned=true lacwm.subgoal_steps=10 lacwm.planning_steps=2 lacwm.n_sg_samples=128 lacwm.subgoal_reached_threshold=10 lacwm.goal_vicinity_threshold=100 lacwm.pretrained_vf=true lacwm.vf_chkpt=output/train/iql_humanoidmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt lacwm.pretrained_policy=true lacwm.policy_chkpt=output/train/iql_humanoidmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

#### HumanoidMaze Large Navigate ####

## IQL ##
python train.py steps=1001000 task=humanoidmaze-large-navigate-v0 gamma=0.999 agent=iql iql.expectile=0.7 iql.alpha=0.4 iql.n_step=5 seed=${SEED}

## TD-MPC ##
python train.py steps=1001000 task=humanoidmaze-large-navigate-v0 agent=tdmpc tdmpc.horizon_plan=25 tdmpc.plan_execute_steps=3 tdmpc.pretrained_vf=true tdmpc.vf_chkpt=output/train/iql_humanoidmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt tdmpc.pretrained_policy=true tdmpc.policy_chkpt=output/train/iql_humanoidmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SAW ##
python train.py steps=1001000 task=humanoidmaze-large-navigate-v0 gamma=0.999 p_trajgoal_policy=1.0 p_randgoal_policy=0.0 agent=saw saw.alpha=3.0 saw.beta=3.0 saw.subgoal_k=100 saw.pretrained_vf=true saw.vf_chkpt=output/train/iql_humanoidmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt saw.pretrained_policy=false saw.pi_extract_type="awr" seed=${SEED}

## HIQL ##
python train.py steps=1001000 task=humanoidmaze-large-navigate-v0 agent=hiql hiql.beta=3.0 hiql.subgoal_k=100 hiql.subgoal_steps=10 hiql.subgoal_reached_threshold=10 hiql.goal_vicinity_threshold=100 hiql.pretrained_vf=true hiql.vf_chkpt=output/train/iql_humanoidmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt hiql.pretrained_policy=true hiql.policy_chkpt=output/train/iql_humanoidmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SD ##
python train.py steps=1001000 task=humanoidmaze-large-navigate-v0 agent=sd sd.goal_conditioned=false sd.n_diffusion_steps=20 sd.ddim_sampling_steps=10 sd.sampling_temperature=1.0 sd.subgoal_k=100 sd.n_sg_samples=128 sd.subgoal_steps=10 sd.subgoal_reached_threshold=10 sd.goal_vicinity_threshold=100 sd.pretrained_vf=true sd.vf_chkpt=output/train/iql_humanoidmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt sd.pretrained_policy=true sd.policy_chkpt=output/train/iql_humanoidmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM ##
python train.py steps=1001000 task=humanoidmaze-large-navigate-v0 agent=telawm telawm.latent_dim=21 telawm.learned_prior=true telawm.beta_kl=0.04 telawm.lam_timescale=100 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.subgoal_state=final telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=100 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_humanoidmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_humanoidmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM Fixed Prior ##
python train.py wandb_name_suffix="_fixedprior" steps=1001000 task=humanoidmaze-large-navigate-v0 agent=telawm telawm.latent_dim=21 telawm.learned_prior=false telawm.beta_kl=0.04 telawm.lam_timescale=100 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=100 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_humanoidmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_humanoidmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt run_suffix=fixedprior seed=${SEED}

## Te-LAWM Single-step ##
python train.py wandb_name_suffix="_singlestep" steps=1001000 task=humanoidmaze-large-navigate-v0 agent=telawm telawm.latent_dim=21 telawm.learned_prior=true telawm.beta_kl=0.04 telawm.lam_timescale=200 telawm.subgoal_steps=10 telawm.planning_steps=1 telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=100 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_humanoidmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_humanoidmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt run_suffix=singlestep seed=${SEED}

## ACWM ##
python train.py steps=1001000 task=humanoidmaze-large-navigate-v0 agent=acwm acwm.wm_timescale=100 acwm.subgoal_steps=10 acwm.planning_steps=2 acwm.n_sg_samples=128 acwm.subgoal_reached_threshold=10 acwm.goal_vicinity_threshold=100 acwm.pretrained_vf=true acwm.vf_chkpt=output/train/iql_humanoidmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt acwm.pretrained_policy=true acwm.policy_chkpt=output/train/iql_humanoidmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## LACWM ##
python train.py steps=1001000 task=humanoidmaze-large-navigate-v0 agent=lacwm lacwm.latent_dim=21 lacwm.learned_prior=true lacwm.beta_kl=0.04 lacwm.wm_timescale=100 lacwm.state_conditioned=true lacwm.subgoal_steps=10 lacwm.planning_steps=2 lacwm.n_sg_samples=128 lacwm.subgoal_reached_threshold=10 lacwm.goal_vicinity_threshold=100 lacwm.pretrained_vf=true lacwm.vf_chkpt=output/train/iql_humanoidmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt lacwm.pretrained_policy=true lacwm.policy_chkpt=output/train/iql_humanoidmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

#### HumanoidMaze Large Stitch ####

## IQL ##
python train.py steps=1001000 task=humanoidmaze-large-stitch-v0 gamma=0.999 agent=iql iql.expectile=0.7 iql.alpha=0.4 iql.n_step=5 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 seed=${SEED}

## TD-MPC ##
python train.py steps=1001000 task=humanoidmaze-large-stitch-v0 agent=tdmpc tdmpc.horizon_plan=25 tdmpc.plan_execute_steps=3 tdmpc.pretrained_vf=true tdmpc.vf_chkpt=output/train/iql_humanoidmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt tdmpc.pretrained_policy=true tdmpc.policy_chkpt=output/train/iql_humanoidmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SAW ##
python train.py steps=1001000 task=humanoidmaze-large-stitch-v0 gamma=0.999 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 agent=saw saw.alpha=3.0 saw.beta=3.0 saw.subgoal_k=100 saw.pretrained_vf=true saw.vf_chkpt=output/train/iql_humanoidmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt saw.pretrained_policy=false saw.pi_extract_type="awr" seed=${SEED}

## HIQL ##
python train.py steps=1001000 task=humanoidmaze-large-stitch-v0 agent=hiql hiql.beta=3.0 hiql.subgoal_k=100 hiql.subgoal_steps=10 hiql.subgoal_reached_threshold=10 hiql.goal_vicinity_threshold=100 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 hiql.pretrained_vf=true hiql.vf_chkpt=output/train/iql_humanoidmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt hiql.pretrained_policy=true hiql.policy_chkpt=output/train/iql_humanoidmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SD ##
python train.py steps=1001000 task=humanoidmaze-large-stitch-v0 agent=sd sd.goal_conditioned=false sd.n_diffusion_steps=20 sd.ddim_sampling_steps=10 sd.sampling_temperature=1.0 sd.subgoal_k=100 sd.n_sg_samples=128 sd.subgoal_steps=10 sd.subgoal_reached_threshold=10 sd.goal_vicinity_threshold=100 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 sd.pretrained_vf=true sd.vf_chkpt=output/train/iql_humanoidmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt sd.pretrained_policy=true sd.policy_chkpt=output/train/iql_humanoidmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM ##   Note: notice that beta_kl=0.04 in all humaze
python train.py steps=1001000 task=humanoidmaze-large-stitch-v0 agent=telawm telawm.latent_dim=21 telawm.learned_prior=true telawm.beta_kl=0.04 telawm.lam_timescale=100 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.subgoal_state=final telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=100 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_humanoidmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_humanoidmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM Fixed Prior ##
python train.py wandb_name_suffix="_fixedprior" steps=1001000 task=humanoidmaze-large-stitch-v0 agent=telawm telawm.latent_dim=21 telawm.learned_prior=false telawm.beta_kl=0.04 telawm.lam_timescale=100 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=100 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_humanoidmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_humanoidmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt run_suffix=fixedprior seed=${SEED}

## Te-LAWM Single-step ##
python train.py wandb_name_suffix="_singlestep" steps=1001000 task=humanoidmaze-large-stitch-v0 agent=telawm telawm.latent_dim=21 telawm.learned_prior=true telawm.beta_kl=0.04 telawm.lam_timescale=200 telawm.subgoal_steps=10 telawm.planning_steps=1 telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=100 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_humanoidmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_humanoidmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt run_suffix=singlestep seed=${SEED}

## ACWM ##
python train.py steps=1001000 task=humanoidmaze-large-stitch-v0 agent=acwm acwm.wm_timescale=100 acwm.subgoal_steps=10 acwm.planning_steps=2 acwm.n_sg_samples=128 acwm.subgoal_reached_threshold=10 acwm.goal_vicinity_threshold=100 acwm.pretrained_vf=true acwm.vf_chkpt=output/train/iql_humanoidmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt acwm.pretrained_policy=true acwm.policy_chkpt=output/train/iql_humanoidmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## LACWM ##
python train.py steps=1001000 task=humanoidmaze-large-stitch-v0 agent=lacwm lacwm.latent_dim=21 lacwm.learned_prior=true lacwm.beta_kl=0.04 lacwm.wm_timescale=100 lacwm.state_conditioned=true lacwm.subgoal_steps=10 lacwm.planning_steps=2 lacwm.n_sg_samples=128 lacwm.subgoal_reached_threshold=10 lacwm.goal_vicinity_threshold=100 lacwm.pretrained_vf=true lacwm.vf_chkpt=output/train/iql_humanoidmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt lacwm.pretrained_policy=true lacwm.policy_chkpt=output/train/iql_humanoidmaze-large-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

#### AntSoccer Arena Navigate ####

## IQL ##
python train.py steps=1001000 task=antsoccer-arena-navigate-v0 gamma=0.99 agent=iql iql.expectile=0.9 iql.alpha=0.2 seed=${SEED}

## TD-MPC ##
python train.py steps=1001000 task=antsoccer-arena-navigate-v0 agent=tdmpc tdmpc.horizon_plan=25 tdmpc.plan_execute_steps=3 tdmpc.pretrained_vf=true tdmpc.vf_chkpt=output/train/iql_antsoccer-arena-navigate-v0_${IQL_SEED}/models/latest.pt tdmpc.pretrained_policy=true tdmpc.policy_chkpt=output/train/iql_antsoccer-arena-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SAW ##
python train.py steps=1001000 task=antsoccer-arena-navigate-v0 gamma=0.99 p_trajgoal_policy=1.0 p_randgoal_policy=0.0 agent=saw saw.alpha=3.0 saw.beta=3.0 saw.subgoal_k=25 saw.pretrained_vf=true saw.vf_chkpt=output/train/iql_antsoccer-arena-navigate-v0_${IQL_SEED}/models/latest.pt saw.pretrained_policy=false saw.pi_extract_type="awr" seed=${SEED}

## HIQL ##
python train.py steps=1001000 task=antsoccer-arena-navigate-v0 agent=hiql hiql.beta=3.0 hiql.subgoal_k=25 hiql.subgoal_steps=10 hiql.subgoal_reached_threshold=10 hiql.goal_vicinity_threshold=70 hiql.pretrained_vf=true hiql.vf_chkpt=output/train/iql_antsoccer-arena-navigate-v0_${IQL_SEED}/models/latest.pt hiql.pretrained_policy=true hiql.policy_chkpt=output/train/iql_antsoccer-arena-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SD ##
python train.py steps=1001000 task=antsoccer-arena-navigate-v0 agent=sd sd.goal_conditioned=false sd.n_diffusion_steps=20 sd.ddim_sampling_steps=10 sd.sampling_temperature=1.0 sd.subgoal_k=25 sd.n_sg_samples=128 sd.subgoal_steps=10 sd.subgoal_reached_threshold=10 sd.goal_vicinity_threshold=70 sd.pretrained_vf=true sd.vf_chkpt=output/train/iql_antsoccer-arena-navigate-v0_${IQL_SEED}/models/latest.pt sd.pretrained_policy=true sd.policy_chkpt=output/train/iql_antsoccer-arena-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM ##
python train.py steps=1001000 task=antsoccer-arena-navigate-v0 agent=telawm telawm.latent_dim=8 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=50 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.subgoal_state=final telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=70 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_antsoccer-arena-navigate-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_antsoccer-arena-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

#### AntSoccer Arena Stitch ####

## IQL ##
python train.py steps=1001000 task=antsoccer-arena-stitch-v0 gamma=0.99 agent=iql iql.expectile=0.9 iql.alpha=0.2 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 seed=${SEED}

## TD-MPC ##
python train.py steps=1001000 task=antsoccer-arena-stitch-v0 agent=tdmpc tdmpc.horizon_plan=25 tdmpc.plan_execute_steps=3 tdmpc.pretrained_vf=true tdmpc.vf_chkpt=output/train/iql_antsoccer-arena-stitch-v0_${IQL_SEED}/models/latest.pt tdmpc.pretrained_policy=true tdmpc.policy_chkpt=output/train/iql_antsoccer-arena-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SAW ##
python train.py steps=1001000 task=antsoccer-arena-stitch-v0 gamma=0.99 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 agent=saw saw.alpha=3.0 saw.beta=3.0 saw.subgoal_k=25 saw.pretrained_vf=true saw.vf_chkpt=output/train/iql_antsoccer-arena-stitch-v0_${IQL_SEED}/models/latest.pt saw.pretrained_policy=false saw.pi_extract_type="awr" seed=${SEED}

## HIQL ##
python train.py steps=1001000 task=antsoccer-arena-stitch-v0 agent=hiql hiql.beta=3.0 hiql.subgoal_k=25 hiql.subgoal_steps=10 hiql.subgoal_reached_threshold=10 hiql.goal_vicinity_threshold=70 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 hiql.pretrained_vf=true hiql.vf_chkpt=output/train/iql_antsoccer-arena-stitch-v0_${IQL_SEED}/models/latest.pt hiql.pretrained_policy=true hiql.policy_chkpt=output/train/iql_antsoccer-arena-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SD ##
python train.py steps=1001000 task=antsoccer-arena-stitch-v0 agent=sd sd.goal_conditioned=false sd.n_diffusion_steps=20 sd.ddim_sampling_steps=10 sd.sampling_temperature=1.0 sd.subgoal_k=25 sd.n_sg_samples=128 sd.subgoal_steps=10 sd.subgoal_reached_threshold=10 sd.goal_vicinity_threshold=70 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 sd.pretrained_vf=true sd.vf_chkpt=output/train/iql_antsoccer-arena-stitch-v0_${IQL_SEED}/models/latest.pt sd.pretrained_policy=true sd.policy_chkpt=output/train/iql_antsoccer-arena-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM ##
python train.py steps=1001000 task=antsoccer-arena-stitch-v0 gamma=0.99 agent=telawm telawm.latent_dim=8 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=50 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.subgoal_state=final telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=70 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_antsoccer-arena-stitch-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_antsoccer-arena-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

#### AntSoccer Medium Navigate ####

## IQL ##
python train.py steps=1001000 task=antsoccer-medium-navigate-v0 gamma=0.99 agent=iql iql.expectile=0.9 iql.alpha=0.5 seed=${SEED}

## TD-MPC ##
python train.py steps=1001000 task=antsoccer-medium-navigate-v0 agent=tdmpc tdmpc.horizon_plan=25 tdmpc.plan_execute_steps=3 tdmpc.pretrained_vf=true tdmpc.vf_chkpt=output/train/iql_antsoccer-medium-navigate-v0_${IQL_SEED}/models/latest.pt tdmpc.pretrained_policy=true tdmpc.policy_chkpt=output/train/iql_antsoccer-medium-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SAW ##
python train.py steps=1001000 task=antsoccer-medium-navigate-v0 gamma=0.99 p_trajgoal_policy=1.0 p_randgoal_policy=0.0 agent=saw saw.alpha=3.0 saw.beta=3.0 saw.subgoal_k=25 saw.pretrained_vf=true saw.vf_chkpt=output/train/iql_antsoccer-medium-navigate-v0_${IQL_SEED}/models/latest.pt saw.pretrained_policy=false saw.pi_extract_type="awr" seed=${SEED}

## HIQL ##
python train.py steps=1001000 task=antsoccer-medium-navigate-v0 gamma=0.99 agent=hiql hiql.beta=3.0 hiql.subgoal_k=25 hiql.subgoal_steps=10 hiql.subgoal_reached_threshold=10 hiql.goal_vicinity_threshold=50 hiql.pretrained_vf=true hiql.vf_chkpt=output/train/iql_antsoccer-medium-navigate-v0_${IQL_SEED}/models/latest.pt hiql.pretrained_policy=true hiql.policy_chkpt=output/train/iql_antsoccer-medium-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SD ##
python train.py steps=1001000 task=antsoccer-medium-navigate-v0 agent=sd sd.goal_conditioned=false sd.n_diffusion_steps=20 sd.ddim_sampling_steps=10 sd.sampling_temperature=1.0 sd.subgoal_k=25 sd.n_sg_samples=128 sd.subgoal_steps=10 sd.subgoal_reached_threshold=10 sd.goal_vicinity_threshold=50 sd.pretrained_vf=true sd.vf_chkpt=output/train/iql_antsoccer-medium-navigate-v0_${IQL_SEED}/models/latest.pt sd.pretrained_policy=true sd.policy_chkpt=output/train/iql_antsoccer-medium-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM ##
python train.py steps=1001000 task=antsoccer-medium-navigate-v0 gamma=0.99 agent=telawm telawm.latent_dim=8 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=100 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=50 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_antsoccer-medium-navigate-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_antsoccer-medium-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM Fixed Prior ##
python train.py wandb_name_suffix="_fixedprior" steps=1001000 task=antsoccer-medium-navigate-v0 gamma=0.99 agent=telawm telawm.latent_dim=8 telawm.learned_prior=false telawm.beta_kl=0.004 telawm.lam_timescale=100 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=50 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_antsoccer-medium-navigate-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_antsoccer-medium-navigate-v0_${IQL_SEED}/models/latest.pt run_suffix=fixedprior seed=${SEED}

## Te-LAWM Single-step ##
python train.py wandb_name_suffix="_singlestep" steps=1001000 task=antsoccer-medium-navigate-v0 gamma=0.99 agent=telawm telawm.latent_dim=8 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=100 telawm.subgoal_steps=10 telawm.planning_steps=1 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=50 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_antsoccer-medium-navigate-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_antsoccer-medium-navigate-v0_${IQL_SEED}/models/latest.pt run_suffix=singlestep seed=${SEED}

## ACWM ##
python train.py steps=1001000 task=antsoccer-medium-navigate-v0 gamma=0.99 agent=acwm acwm.wm_timescale=100 acwm.subgoal_steps=10 acwm.planning_steps=2 acwm.subgoal_state=first acwm.n_sg_samples=128 acwm.subgoal_reached_threshold=10 acwm.goal_vicinity_threshold=50 acwm.pretrained_vf=true acwm.vf_chkpt=output/train/iql_antsoccer-medium-navigate-v0_${IQL_SEED}/models/latest.pt acwm.pretrained_policy=true acwm.policy_chkpt=output/train/iql_antsoccer-medium-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## LACWM ##
python train.py steps=1001000 task=antsoccer-medium-navigate-v0 gamma=0.99 agent=lacwm lacwm.latent_dim=8 lacwm.learned_prior=true lacwm.beta_kl=0.004 lacwm.wm_timescale=100 lacwm.state_conditioned=true lacwm.subgoal_steps=10 lacwm.planning_steps=2 lacwm.subgoal_state=first lacwm.n_sg_samples=128 lacwm.subgoal_reached_threshold=10 lacwm.goal_vicinity_threshold=50 lacwm.pretrained_vf=true lacwm.vf_chkpt=output/train/iql_antsoccer-medium-navigate-v0_${IQL_SEED}/models/latest.pt lacwm.pretrained_policy=true lacwm.policy_chkpt=output/train/iql_antsoccer-medium-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

#### Cube Single Play ####

## IQL ##
python train.py steps=1001000 task=cube-single-play-v0 agent=iql iql.expectile=0.9 iql.alpha=3.0 seed=${SEED}

## TD-MPC ##
python train.py steps=1001000 task=cube-single-play-v0 agent=tdmpc tdmpc.horizon_plan=25 tdmpc.plan_execute_steps=3 tdmpc.pretrained_vf=true tdmpc.vf_chkpt=output/train/iql_cube-single-play-v0_${IQL_SEED}/models/latest.pt tdmpc.pretrained_policy=true tdmpc.policy_chkpt=output/train/iql_cube-single-play-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SAW ##
python train.py steps=1001000 task=cube-single-play-v0 gamma=0.99 p_trajgoal_policy=1.0 p_randgoal_policy=0.0 agent=saw saw.alpha=3.0 saw.beta=0.3 saw.subgoal_k=10 saw.pretrained_vf=true saw.vf_chkpt=output/train/iql_cube-single-play-v0_${IQL_SEED}/models/latest.pt saw.pretrained_policy=false saw.pi_extract_type="awr" seed=${SEED}

## HIQL ##
python train.py steps=1001000 task=cube-single-play-v0 agent=hiql hiql.beta=3.0 hiql.subgoal_k=10 hiql.subgoal_steps=10 hiql.subgoal_reached_threshold=0 hiql.goal_vicinity_threshold=10 hiql.pretrained_vf=true hiql.vf_chkpt=output/train/iql_cube-single-play-v0_${IQL_SEED}/models/latest.pt hiql.pretrained_policy=true hiql.policy_chkpt=output/train/iql_cube-single-play-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SD ##
python train.py steps=1001000 task=cube-single-play-v0 agent=sd sd.goal_conditioned=false sd.n_diffusion_steps=20 sd.ddim_sampling_steps=10 sd.sampling_temperature=1.0 sd.subgoal_k=10 sd.n_sg_samples=1024 sd.subgoal_steps=10 sd.subgoal_reached_threshold=0 sd.goal_vicinity_threshold=10 sd.pretrained_vf=true sd.vf_chkpt=output/train/iql_cube-single-play-v0_${IQL_SEED}/models/latest.pt sd.pretrained_policy=true sd.policy_chkpt=output/train/iql_cube-single-play-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM ##
python train.py steps=1001000 task=cube-single-play-v0 agent=telawm telawm.latent_dim=5 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=10 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.subgoal_state=first telawm.n_sg_samples=1024 telawm.subgoal_reached_threshold=0 telawm.goal_vicinity_threshold=10 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_cube-single-play-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_cube-single-play-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

#### Cube Single Noisy ####

## IQL ##
python train.py steps=1001000 task=cube-single-noisy-v0 agent=iql iql.expectile=0.9 iql.alpha=0.1 seed=${SEED}

## TD-MPC ##
python train.py steps=1001000 task=cube-single-noisy-v0 agent=tdmpc tdmpc.horizon_plan=25 tdmpc.plan_execute_steps=3 tdmpc.pretrained_vf=true tdmpc.vf_chkpt=output/train/iql_cube-single-noisy-v0_${IQL_SEED}/models/latest.pt tdmpc.pretrained_policy=true tdmpc.policy_chkpt=output/train/iql_cube-single-noisy-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SAW ##
python train.py steps=1001000 task=cube-single-noisy-v0 gamma=0.99 p_trajgoal_policy=1.0 p_randgoal_policy=0.0 agent=saw saw.alpha=3.0 saw.beta=0.3 saw.subgoal_k=10 saw.pretrained_vf=true saw.vf_chkpt=output/train/iql_cube-single-noisy-v0_${IQL_SEED}/models/latest.pt saw.pretrained_policy=false saw.pi_extract_type="awr" seed=${SEED}

## HIQL ##
python train.py steps=1001000 task=cube-single-noisy-v0 agent=hiql hiql.beta=3.0 hiql.subgoal_k=10 hiql.subgoal_steps=10 hiql.subgoal_reached_threshold=0 hiql.goal_vicinity_threshold=10 hiql.pretrained_vf=true hiql.vf_chkpt=output/train/iql_cube-single-noisy-v0_${IQL_SEED}/models/latest.pt hiql.pretrained_policy=true hiql.policy_chkpt=output/train/iql_cube-single-noisy-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SD ##
python train.py steps=1001000 task=cube-single-noisy-v0 agent=sd sd.goal_conditioned=false sd.n_diffusion_steps=20 sd.ddim_sampling_steps=10 sd.sampling_temperature=1.0 sd.subgoal_k=10 sd.n_sg_samples=1024 sd.subgoal_steps=10 sd.subgoal_reached_threshold=0 sd.goal_vicinity_threshold=10 sd.pretrained_vf=true sd.vf_chkpt=output/train/iql_cube-single-noisy-v0_${IQL_SEED}/models/latest.pt sd.pretrained_policy=true sd.policy_chkpt=output/train/iql_cube-single-noisy-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM ##
python train.py steps=1001000 task=cube-single-noisy-v0 agent=telawm telawm.latent_dim=5 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=10 telawm.subgoal_steps=10 telawm.planning_steps=8 telawm.subgoal_state=final telawm.n_sg_samples=1024 telawm.subgoal_reached_threshold=0 telawm.goal_vicinity_threshold=10 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_cube-single-noisy-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_cube-single-noisy-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM Fixed Prior ##
python train.py wandb_name_suffix="_fixedprior" steps=1001000 task=cube-single-noisy-v0 agent=telawm telawm.latent_dim=5 telawm.learned_prior=false telawm.beta_kl=0.004 telawm.lam_timescale=10 telawm.subgoal_steps=10 telawm.planning_steps=8 telawm.n_sg_samples=1024 telawm.subgoal_reached_threshold=0 telawm.goal_vicinity_threshold=10 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_cube-single-noisy-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_cube-single-noisy-v0_${IQL_SEED}/models/latest.pt run_suffix=fixedprior seed=${SEED}

## Te-LAWM Single-step ##
python train.py wandb_name_suffix="_singlestep" steps=1001000 task=cube-single-noisy-v0 agent=telawm telawm.latent_dim=5 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=20 telawm.subgoal_steps=10 telawm.planning_steps=1 telawm.n_sg_samples=1024 telawm.subgoal_reached_threshold=0 telawm.goal_vicinity_threshold=10 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_cube-single-noisy-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_cube-single-noisy-v0_${IQL_SEED}/models/latest.pt run_suffix=singlestep seed=${SEED}

## ACWM ##
python train.py steps=1001000 task=cube-single-noisy-v0 agent=acwm acwm.wm_timescale=10 acwm.subgoal_steps=10 acwm.planning_steps=8 acwm.n_sg_samples=1024 acwm.subgoal_reached_threshold=0 acwm.goal_vicinity_threshold=10 acwm.pretrained_vf=true acwm.vf_chkpt=output/train/iql_cube-single-noisy-v0_${IQL_SEED}/models/latest.pt acwm.pretrained_policy=true acwm.policy_chkpt=output/train/iql_cube-single-noisy-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## LACWM ##
python train.py steps=1001000 task=cube-single-noisy-v0 agent=lacwm lacwm.latent_dim=5 lacwm.learned_prior=true lacwm.beta_kl=0.004 lacwm.wm_timescale=10 lacwm.state_conditioned=true lacwm.subgoal_steps=10 lacwm.planning_steps=8 lacwm.n_sg_samples=1024 lacwm.subgoal_reached_threshold=0 lacwm.goal_vicinity_threshold=10 lacwm.pretrained_vf=true lacwm.vf_chkpt=output/train/iql_cube-single-noisy-v0_${IQL_SEED}/models/latest.pt lacwm.pretrained_policy=true lacwm.policy_chkpt=output/train/iql_cube-single-noisy-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

#### Cube Double Play ####

## IQL ##
python train.py steps=1001000 task=cube-double-play-v0 agent=iql iql.expectile=0.9 iql.alpha=3.0 seed=${SEED}

## TD-MPC ##
python train.py steps=1001000 task=cube-double-play-v0 agent=tdmpc tdmpc.horizon_plan=25 tdmpc.plan_execute_steps=3 tdmpc.pretrained_vf=true tdmpc.vf_chkpt=output/train/iql_cube-double-play-v0_${IQL_SEED}/models/latest.pt tdmpc.pretrained_policy=true tdmpc.policy_chkpt=output/train/iql_cube-double-play-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SAW ##
python train.py steps=1001000 task=cube-double-play-v0 gamma=0.99 p_trajgoal_policy=1.0 p_randgoal_policy=0.0 agent=saw saw.alpha=3.0 saw.beta=1.0 saw.subgoal_k=25 saw.pretrained_vf=true saw.vf_chkpt=output/train/iql_cube-double-play-v0_${IQL_SEED}/models/latest.pt saw.pretrained_policy=false saw.pi_extract_type="awr" seed=${SEED}

## HIQL ##
python train.py steps=1001000 task=cube-double-play-v0 agent=hiql hiql.beta=3.0 hiql.subgoal_k=10 hiql.subgoal_steps=10 hiql.subgoal_reached_threshold=0 hiql.goal_vicinity_threshold=30 hiql.pretrained_vf=true hiql.vf_chkpt=output/train/iql_cube-double-play-v0_${IQL_SEED}/models/latest.pt hiql.pretrained_policy=true hiql.policy_chkpt=output/train/iql_cube-double-play-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SD ##
python train.py steps=1001000 task=cube-double-play-v0 agent=sd sd.goal_conditioned=false sd.n_diffusion_steps=20 sd.ddim_sampling_steps=10 sd.sampling_temperature=1.0 sd.subgoal_k=50 sd.n_sg_samples=1024 sd.subgoal_steps=50 sd.subgoal_reached_threshold=0 sd.goal_vicinity_threshold=30 sd.pretrained_vf=true sd.vf_chkpt=output/train/iql_cube-double-play-v0_${IQL_SEED}/models/latest.pt sd.pretrained_policy=true sd.policy_chkpt=output/train/iql_cube-double-play-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM ##
python train.py steps=1001000 task=cube-double-play-v0 agent=telawm telawm.latent_dim=5 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=50 telawm.subgoal_steps=50 telawm.planning_steps=2 telawm.subgoal_state=final telawm.n_sg_samples=1024 telawm.subgoal_reached_threshold=0 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_cube-double-play-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_cube-double-play-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM Fixed Prior ##
python train.py wandb_name_suffix="_fixedprior" steps=1001000 task=cube-double-play-v0 agent=telawm telawm.latent_dim=5 telawm.learned_prior=false telawm.beta_kl=0.004 telawm.lam_timescale=50 telawm.subgoal_steps=50 telawm.planning_steps=2 telawm.n_sg_samples=1024 telawm.subgoal_reached_threshold=0 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_cube-double-play-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_cube-double-play-v0_${IQL_SEED}/models/latest.pt run_suffix=fixedprior seed=${SEED}

## Te-LAWM Single-step ##
python train.py wandb_name_suffix="_singlestep" steps=1001000 task=cube-double-play-v0 agent=telawm telawm.latent_dim=5 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=100 telawm.subgoal_steps=50 telawm.planning_steps=1 telawm.n_sg_samples=1024 telawm.subgoal_reached_threshold=0 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_cube-double-play-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_cube-double-play-v0_${IQL_SEED}/models/latest.pt run_suffix=singlestep seed=${SEED}

## ACWM ##
python train.py steps=1001000 task=cube-double-play-v0 agent=acwm acwm.wm_timescale=50 acwm.subgoal_steps=50 acwm.planning_steps=2 acwm.n_sg_samples=1024 acwm.subgoal_reached_threshold=0 acwm.goal_vicinity_threshold=30 acwm.pretrained_vf=true acwm.vf_chkpt=output/train/iql_cube-double-play-v0_${IQL_SEED}/models/latest.pt acwm.pretrained_policy=true acwm.policy_chkpt=output/train/iql_cube-double-play-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## LACWM ##
python train.py steps=1001000 task=cube-double-play-v0 agent=lacwm lacwm.latent_dim=5 lacwm.learned_prior=true lacwm.beta_kl=0.004 lacwm.wm_timescale=50 lacwm.state_conditioned=true lacwm.subgoal_steps=50 lacwm.planning_steps=2 lacwm.n_sg_samples=1024 lacwm.subgoal_reached_threshold=0 lacwm.goal_vicinity_threshold=30 lacwm.pretrained_vf=true lacwm.vf_chkpt=output/train/iql_cube-double-play-v0_${IQL_SEED}/models/latest.pt lacwm.pretrained_policy=true lacwm.policy_chkpt=output/train/iql_cube-double-play-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

#### Cube Double Noisy ####

## IQL ##
python train.py steps=1001000 task=cube-double-noisy-v0 agent=iql iql.expectile=0.9 iql.alpha=0.1 seed=${SEED}

## TD-MPC ##
python train.py steps=1001000 task=cube-double-noisy-v0 agent=tdmpc tdmpc.horizon_plan=25 tdmpc.plan_execute_steps=3 tdmpc.pretrained_vf=true tdmpc.vf_chkpt=output/train/iql_cube-double-noisy-v0_${IQL_SEED}/models/latest.pt tdmpc.pretrained_policy=true tdmpc.policy_chkpt=output/train/iql_cube-double-noisy-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SAW ##
python train.py steps=1001000 task=cube-double-noisy-v0 gamma=0.99 p_trajgoal_policy=1.0 p_randgoal_policy=0.0 agent=saw saw.alpha=3.0 saw.beta=1.0 saw.subgoal_k=10 saw.pretrained_vf=true saw.vf_chkpt=output/train/iql_cube-double-noisy-v0_${IQL_SEED}/models/latest.pt saw.pretrained_policy=false saw.pi_extract_type="awr" seed=${SEED}

## HIQL ##
python train.py steps=1001000 task=cube-double-noisy-v0 agent=hiql hiql.beta=3.0 hiql.subgoal_k=10 hiql.subgoal_steps=10 hiql.subgoal_reached_threshold=0 hiql.goal_vicinity_threshold=30 hiql.pretrained_vf=true hiql.vf_chkpt=output/train/iql_cube-double-noisy-v0_${IQL_SEED}/models/latest.pt hiql.pretrained_policy=true hiql.policy_chkpt=output/train/iql_cube-double-noisy-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SD ##
python train.py steps=1001000 task=cube-double-noisy-v0 agent=sd sd.goal_conditioned=false sd.n_diffusion_steps=20 sd.ddim_sampling_steps=10 sd.sampling_temperature=1.0 sd.subgoal_k=10 sd.n_sg_samples=1024 sd.subgoal_steps=10 sd.subgoal_reached_threshold=0 sd.goal_vicinity_threshold=30 sd.pretrained_vf=true sd.vf_chkpt=output/train/iql_cube-double-noisy-v0_${IQL_SEED}/models/latest.pt sd.pretrained_policy=true sd.policy_chkpt=output/train/iql_cube-double-noisy-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM ##
python train.py steps=1001000 task=cube-double-noisy-v0 agent=telawm telawm.latent_dim=5 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=10 telawm.subgoal_steps=10 telawm.planning_steps=8 telawm.subgoal_state=final telawm.n_sg_samples=1024 telawm.subgoal_reached_threshold=0 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_cube-double-noisy-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_cube-double-noisy-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM Fixed Prior ##
python train.py wandb_name_suffix="_fixedprior" steps=1001000 task=cube-double-noisy-v0 agent=telawm telawm.latent_dim=5 telawm.learned_prior=false telawm.beta_kl=0.004 telawm.lam_timescale=10 telawm.subgoal_steps=10 telawm.planning_steps=8 telawm.n_sg_samples=1024 telawm.subgoal_reached_threshold=0 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_cube-double-noisy-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_cube-double-noisy-v0_${IQL_SEED}/models/latest.pt run_suffix=fixedprior seed=${SEED}

## Te-LAWM Single-step ##
python train.py wandb_name_suffix="_singlestep" steps=1001000 task=cube-double-noisy-v0 agent=telawm telawm.latent_dim=5 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=50 telawm.subgoal_steps=10 telawm.planning_steps=1 telawm.n_sg_samples=1024 telawm.subgoal_reached_threshold=0 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_cube-double-noisy-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_cube-double-noisy-v0_${IQL_SEED}/models/latest.pt run_suffix=singlestep seed=${SEED}

## ACWM ##
python train.py steps=1001000 task=cube-double-noisy-v0 agent=acwm acwm.wm_timescale=10 acwm.subgoal_steps=10 acwm.planning_steps=8 acwm.n_sg_samples=1024 acwm.subgoal_reached_threshold=0 acwm.goal_vicinity_threshold=30 acwm.pretrained_vf=true acwm.vf_chkpt=output/train/iql_cube-double-noisy-v0_${IQL_SEED}/models/latest.pt acwm.pretrained_policy=true acwm.policy_chkpt=output/train/iql_cube-double-noisy-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## LACWM ##
python train.py steps=1001000 task=cube-double-noisy-v0 agent=lacwm lacwm.latent_dim=5 lacwm.learned_prior=true lacwm.beta_kl=0.004 lacwm.wm_timescale=10 lacwm.state_conditioned=true lacwm.subgoal_steps=10 lacwm.planning_steps=8 lacwm.n_sg_samples=1024 lacwm.subgoal_reached_threshold=0 lacwm.goal_vicinity_threshold=30 lacwm.pretrained_vf=true lacwm.vf_chkpt=output/train/iql_cube-double-noisy-v0_${IQL_SEED}/models/latest.pt lacwm.pretrained_policy=true lacwm.policy_chkpt=output/train/iql_cube-double-noisy-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

#### Cube Triple Play ####

## IQL ##
python train.py steps=1001000 task=cube-triple-play-v0 agent=iql iql.expectile=0.9 iql.alpha=3.0 seed=${SEED}

## TD-MPC ##
python train.py steps=1001000 task=cube-triple-play-v0 agent=tdmpc tdmpc.horizon_plan=25 tdmpc.plan_execute_steps=3 tdmpc.pretrained_vf=true tdmpc.vf_chkpt=output/train/iql_cube-triple-play-v0_${IQL_SEED}/models/latest.pt tdmpc.pretrained_policy=true tdmpc.policy_chkpt=output/train/iql_cube-triple-play-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SAW ##
python train.py steps=1001000 task=cube-triple-play-v0 gamma=0.99 p_trajgoal_policy=1.0 p_randgoal_policy=0.0 agent=saw saw.alpha=3.0 saw.beta=1.0 saw.subgoal_k=10 saw.pretrained_vf=true saw.vf_chkpt=output/train/iql_cube-triple-play-v0_${IQL_SEED}/models/latest.pt saw.pretrained_policy=false saw.pi_extract_type="awr" seed=${SEED}

## HIQL ##
python train.py steps=1001000 task=cube-triple-play-v0 agent=hiql hiql.beta=3.0 hiql.subgoal_k=10 hiql.subgoal_steps=10 hiql.subgoal_reached_threshold=0 hiql.goal_vicinity_threshold=30 hiql.pretrained_vf=true hiql.vf_chkpt=output/train/iql_cube-triple-play-v0_${IQL_SEED}/models/latest.pt hiql.pretrained_policy=true hiql.policy_chkpt=output/train/iql_cube-triple-play-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SD ##
python train.py steps=1001000 task=cube-triple-play-v0 agent=sd sd.goal_conditioned=false sd.n_diffusion_steps=20 sd.ddim_sampling_steps=10 sd.sampling_temperature=1.0 sd.subgoal_k=50 sd.n_sg_samples=1024 sd.subgoal_steps=50 sd.subgoal_reached_threshold=0 sd.goal_vicinity_threshold=30 sd.pretrained_vf=true sd.vf_chkpt=output/train/iql_cube-triple-play-v0_${IQL_SEED}/models/latest.pt sd.pretrained_policy=true sd.policy_chkpt=output/train/iql_cube-triple-play-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM ##
python train.py steps=1001000 task=cube-triple-play-v0 agent=telawm telawm.latent_dim=5 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=50 telawm.subgoal_steps=50 telawm.planning_steps=2 telawm.subgoal_state=first telawm.n_sg_samples=1024 telawm.subgoal_reached_threshold=0 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_cube-triple-play-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_cube-triple-play-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

#### Cube Triple Noisy ####

## IQL ##
python train.py steps=1001000 task=cube-triple-noisy-v0 agent=iql iql.expectile=0.9 iql.alpha=0.1 seed=${SEED}

## TD-MPC ##
python train.py steps=1001000 task=cube-triple-noisy-v0 agent=tdmpc tdmpc.horizon_plan=25 tdmpc.plan_execute_steps=3 tdmpc.pretrained_vf=true tdmpc.vf_chkpt=output/train/iql_cube-triple-noisy-v0_${IQL_SEED}/models/latest.pt tdmpc.pretrained_policy=true tdmpc.policy_chkpt=output/train/iql_cube-triple-noisy-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SAW ##
python train.py steps=1001000 task=cube-triple-noisy-v0 gamma=0.99 p_trajgoal_policy=1.0 p_randgoal_policy=0.0 agent=saw saw.alpha=3.0 saw.beta=1.0 saw.subgoal_k=10 saw.pretrained_vf=true saw.vf_chkpt=output/train/iql_cube-triple-noisy-v0_${IQL_SEED}/models/latest.pt saw.pretrained_policy=false saw.pi_extract_type="awr" seed=${SEED}

## HIQL ##
python train.py steps=1001000 task=cube-triple-noisy-v0 agent=hiql hiql.beta=3.0 hiql.subgoal_k=10 hiql.subgoal_steps=10 hiql.subgoal_reached_threshold=0 hiql.goal_vicinity_threshold=30 hiql.pretrained_vf=true hiql.vf_chkpt=output/train/iql_cube-triple-noisy-v0_${IQL_SEED}/models/latest.pt hiql.pretrained_policy=true hiql.policy_chkpt=output/train/iql_cube-triple-noisy-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SD ##
python train.py steps=1001000 task=cube-triple-noisy-v0 agent=sd sd.goal_conditioned=false sd.n_diffusion_steps=20 sd.ddim_sampling_steps=10 sd.sampling_temperature=1.0 sd.subgoal_k=10 sd.n_sg_samples=1024 sd.subgoal_steps=10 sd.subgoal_reached_threshold=0 sd.goal_vicinity_threshold=30 sd.pretrained_vf=true sd.vf_chkpt=output/train/iql_cube-triple-noisy-v0_${IQL_SEED}/models/latest.pt sd.pretrained_policy=true sd.policy_chkpt=output/train/iql_cube-triple-noisy-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM ##
python train.py steps=1001000 task=cube-triple-noisy-v0 agent=telawm telawm.latent_dim=5 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=10 telawm.subgoal_steps=10 telawm.planning_steps=8 telawm.subgoal_state=final telawm.n_sg_samples=1024 telawm.subgoal_reached_threshold=0 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_cube-triple-noisy-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_cube-triple-noisy-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM Fixed Prior ##
python train.py wandb_name_suffix="_fixedprior" steps=1001000 task=cube-triple-noisy-v0 agent=telawm telawm.latent_dim=5 telawm.learned_prior=false telawm.beta_kl=0.004 telawm.lam_timescale=10 telawm.subgoal_steps=10 telawm.planning_steps=8 telawm.n_sg_samples=1024 telawm.subgoal_reached_threshold=0 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_cube-triple-noisy-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_cube-triple-noisy-v0_${IQL_SEED}/models/latest.pt run_suffix=fixedprior seed=${SEED}

## Te-LAWM Single-step ##
python train.py wandb_name_suffix="_singlestep" steps=1001000 task=cube-triple-noisy-v0 agent=telawm telawm.latent_dim=5 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=50 telawm.subgoal_steps=10 telawm.planning_steps=1 telawm.n_sg_samples=1024 telawm.subgoal_reached_threshold=0 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_cube-triple-noisy-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_cube-triple-noisy-v0_${IQL_SEED}/models/latest.pt run_suffix=singlestep seed=${SEED}

## ACWM ##
python train.py steps=1001000 task=cube-triple-noisy-v0 agent=acwm acwm.wm_timescale=10 acwm.subgoal_steps=10 acwm.planning_steps=8 acwm.n_sg_samples=1024 acwm.subgoal_reached_threshold=0 acwm.goal_vicinity_threshold=30 acwm.pretrained_vf=true acwm.vf_chkpt=output/train/iql_cube-triple-noisy-v0_${IQL_SEED}/models/latest.pt acwm.pretrained_policy=true acwm.policy_chkpt=output/train/iql_cube-triple-noisy-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## LACWM ##
python train.py steps=1001000 task=cube-triple-noisy-v0 agent=lacwm lacwm.latent_dim=5 lacwm.learned_prior=true lacwm.beta_kl=0.004 lacwm.wm_timescale=10 lacwm.state_conditioned=true lacwm.subgoal_steps=10 lacwm.planning_steps=8 lacwm.n_sg_samples=1024 lacwm.subgoal_reached_threshold=0 lacwm.goal_vicinity_threshold=30 lacwm.pretrained_vf=true lacwm.vf_chkpt=output/train/iql_cube-triple-noisy-v0_${IQL_SEED}/models/latest.pt lacwm.pretrained_policy=true lacwm.policy_chkpt=output/train/iql_cube-triple-noisy-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

#### Scene Play ####

## IQL ##
python train.py steps=1001000 task=scene-play-v0 agent=iql iql.expectile=0.9 iql.alpha=3.0 seed=${SEED}

## TD-MPC ##
python train.py steps=1001000 task=scene-play-v0 agent=tdmpc tdmpc.horizon_plan=25 tdmpc.plan_execute_steps=3 tdmpc.pretrained_vf=true tdmpc.vf_chkpt=output/train/iql_scene-play-v0_${IQL_SEED}/models/latest.pt tdmpc.pretrained_policy=true tdmpc.policy_chkpt=output/train/iql_scene-play-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SAW ##
python train.py steps=1001000 task=scene-play-v0 gamma=0.99 p_trajgoal_policy=1.0 p_randgoal_policy=0.0 agent=saw saw.alpha=3.0 saw.beta=1.0 saw.subgoal_k=10 saw.pretrained_vf=true saw.vf_chkpt=output/train/iql_scene-play-v0_${IQL_SEED}/models/latest.pt saw.pretrained_policy=false saw.pi_extract_type="awr" seed=${SEED}

## HIQL ##
python train.py steps=1001000 task=scene-play-v0 agent=hiql hiql.beta=3.0 hiql.subgoal_k=25 hiql.subgoal_steps=10 hiql.subgoal_reached_threshold=1 hiql.goal_vicinity_threshold=30 hiql.pretrained_vf=true hiql.vf_chkpt=output/train/iql_scene-play-v0_${IQL_SEED}/models/latest.pt hiql.pretrained_policy=true hiql.policy_chkpt=output/train/iql_scene-play-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SD ##
python train.py steps=1001000 task=scene-play-v0 agent=sd sd.goal_conditioned=false sd.n_diffusion_steps=20 sd.ddim_sampling_steps=10 sd.sampling_temperature=1.0 sd.subgoal_k=50 sd.n_sg_samples=128 sd.subgoal_steps=50 sd.subgoal_reached_threshold=1 sd.goal_vicinity_threshold=30 sd.pretrained_vf=true sd.vf_chkpt=output/train/iql_scene-play-v0_${IQL_SEED}/models/latest.pt sd.pretrained_policy=true sd.policy_chkpt=output/train/iql_scene-play-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM ##
python train.py steps=1001000 task=scene-play-v0 agent=telawm telawm.latent_dim=5 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=50 telawm.subgoal_steps=50 telawm.planning_steps=2 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=1 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_scene-play-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_scene-play-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM Fixed Prior ##
python train.py wandb_name_suffix="_fixedprior" steps=1001000 task=scene-play-v0 agent=telawm telawm.latent_dim=5 telawm.learned_prior=false telawm.beta_kl=0.004 telawm.lam_timescale=50 telawm.subgoal_steps=50 telawm.planning_steps=2 telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=1 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_scene-play-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_scene-play-v0_${IQL_SEED}/models/latest.pt run_suffix=fixedprior seed=${SEED}

## Te-LAWM Single-step ##
python train.py wandb_name_suffix="_singlestep" steps=1001000 task=scene-play-v0 agent=telawm telawm.latent_dim=5 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=100 telawm.subgoal_steps=50 telawm.planning_steps=1 telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=1 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_scene-play-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_scene-play-v0_${IQL_SEED}/models/latest.pt run_suffix=singlestep seed=${SEED}

## ACWM ##
python train.py steps=1001000 task=scene-play-v0 agent=acwm acwm.wm_timescale=50 acwm.subgoal_steps=50 acwm.planning_steps=2 acwm.n_sg_samples=128 acwm.subgoal_reached_threshold=1 acwm.goal_vicinity_threshold=30 acwm.pretrained_vf=true acwm.vf_chkpt=output/train/iql_scene-play-v0_${IQL_SEED}/models/latest.pt acwm.pretrained_policy=true acwm.policy_chkpt=output/train/iql_scene-play-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## LACWM ##
python train.py steps=1001000 task=scene-play-v0 agent=lacwm lacwm.latent_dim=5 lacwm.learned_prior=true lacwm.beta_kl=0.004 lacwm.wm_timescale=50 lacwm.state_conditioned=true lacwm.subgoal_steps=50 lacwm.planning_steps=2 lacwm.n_sg_samples=128 lacwm.subgoal_reached_threshold=1 lacwm.goal_vicinity_threshold=30 lacwm.pretrained_vf=true lacwm.vf_chkpt=output/train/iql_scene-play-v0_${IQL_SEED}/models/latest.pt lacwm.pretrained_policy=true lacwm.policy_chkpt=output/train/iql_scene-play-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

#### Scene Noisy ####

## IQL ##
python train.py steps=1001000 task=scene-noisy-v0 agent=iql iql.expectile=0.9 iql.alpha=0.1 seed=${SEED}

## TD-MPC ##
python train.py steps=1001000 task=scene-noisy-v0 agent=tdmpc tdmpc.horizon_plan=25 tdmpc.plan_execute_steps=3 tdmpc.pretrained_vf=true tdmpc.vf_chkpt=output/train/iql_scene-noisy-v0_${IQL_SEED}/models/latest.pt tdmpc.pretrained_policy=true tdmpc.policy_chkpt=output/train/iql_scene-noisy-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SAW ##
python train.py steps=1001000 task=scene-noisy-v0 gamma=0.99 p_trajgoal_policy=1.0 p_randgoal_policy=0.0 agent=saw saw.alpha=3.0 saw.beta=1.0 saw.subgoal_k=10 saw.pretrained_vf=true saw.vf_chkpt=output/train/iql_scene-noisy-v0_${IQL_SEED}/models/latest.pt saw.pretrained_policy=false saw.pi_extract_type="awr" seed=${SEED}

## HIQL ##
python train.py steps=1001000 task=scene-noisy-v0 agent=hiql hiql.beta=3.0 hiql.subgoal_k=50 hiql.subgoal_steps=10 hiql.subgoal_reached_threshold=1 hiql.goal_vicinity_threshold=30 hiql.pretrained_vf=true hiql.vf_chkpt=output/train/iql_scene-noisy-v0_${IQL_SEED}/models/latest.pt hiql.pretrained_policy=true hiql.policy_chkpt=output/train/iql_scene-noisy-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SD ##
python train.py steps=1001000 task=scene-noisy-v0 agent=sd sd.goal_conditioned=false sd.n_diffusion_steps=20 sd.ddim_sampling_steps=10 sd.sampling_temperature=1.0 sd.subgoal_k=50 sd.n_sg_samples=128 sd.subgoal_steps=10 sd.subgoal_reached_threshold=1 sd.goal_vicinity_threshold=30 sd.pretrained_vf=true sd.vf_chkpt=output/train/iql_scene-noisy-v0_${IQL_SEED}/models/latest.pt sd.pretrained_policy=true sd.policy_chkpt=output/train/iql_scene-noisy-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## Te-LAWM ##
python train.py steps=1001000 task=scene-noisy-v0 agent=telawm telawm.latent_dim=5 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=50 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.subgoal_state=first telawm.n_sg_samples=1024 telawm.subgoal_reached_threshold=1 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_scene-noisy-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_scene-noisy-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

#### Visual AntMaze Medium Navigate ####

## AE ##
python train.py steps=200000 task=visual-antmaze-medium-navigate-v0 obs=rgb batch_size=256 eval_freq=25000 agent=ae seed=${SEED}

## IQL ##
python train.py steps=501000 task=visual-antmaze-medium-navigate-v0 obs=rgb batch_size=256 gamma=0.99 eval_freq=250000 agent=iql iql.expectile=0.7 iql.alpha=0.1 seed=${SEED}

## HIQL ##
python train.py steps=1001000 task=visual-antmaze-medium-navigate-v0 obs=rgb batch_size=256 gamma=0.99 eval_freq=250000 agent=hiql latent_encoder_checkpoint=output/train/ae_visual-antmaze-medium-navigate-v0_${AE_SEED}/models/latest.pt hiql.beta=3.0 hiql.subgoal_k=25 hiql.subgoal_steps=10 hiql.subgoal_reached_threshold=10 hiql.goal_vicinity_threshold=30 hiql.pretrained_vf=true hiql.vf_chkpt=output/train/iql_visual-antmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt hiql.pretrained_policy=true hiql.policy_chkpt=output/train/iql_visual-antmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SD ##
python train.py steps=1001000 task=visual-antmaze-medium-navigate-v0 obs=latent latent_encoder_checkpoint=output/train/ae_visual-antmaze-medium-navigate-v0_${AE_SEED}/models/latest.pt gamma=0.99 agent=sd sd.goal_conditioned=false sd.n_diffusion_steps=20 sd.ddim_sampling_steps=10 sd.sampling_temperature=1.0 sd.subgoal_k=25 sd.n_sg_samples=128 sd.subgoal_steps=10 sd.subgoal_reached_threshold=10 sd.goal_vicinity_threshold=30 sd.pretrained_vf=false sd.pretrained_policy=false seed=${SEED}
python evaluate.py task=visual-antmaze-medium-navigate-v0 obs=rgb latent_encoder_checkpoint=output/train/ae_visual-antmaze-medium-navigate-v0_${AE_SEED}/models/latest.pt gamma=0.99 agent=sd sd.goal_conditioned=false sd.n_diffusion_steps=20 sd.ddim_sampling_steps=10 sd.sampling_temperature=1.0 sd.subgoal_k=25 sd.n_sg_samples=128 sd.subgoal_steps=10 sd.subgoal_reached_threshold=10 sd.goal_vicinity_threshold=30 sd.pretrained_vf=true sd.vf_chkpt=output/train/iql_visual-antmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt sd.pretrained_policy=true sd.policy_chkpt=output/train/iql_visual-antmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt eval_checkpoint=output/train/sd_visual-antmaze-medium-navigate-v0_${SEED}/models/latest.pt seed=${SEED}

## Te-LAWM ##
python train.py steps=1001000 task=visual-antmaze-medium-navigate-v0 obs=latent latent_encoder_checkpoint=output/train/ae_visual-antmaze-medium-navigate-v0_${AE_SEED}/models/latest.pt gamma=0.99 agent=telawm telawm.latent_dim=8 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=25 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=30 seed=${SEED}
python evaluate.py task=visual-antmaze-medium-navigate-v0 obs=rgb latent_encoder_checkpoint=output/train/ae_visual-antmaze-medium-navigate-v0_${AE_SEED}/models/latest.pt gamma=0.99 agent=telawm telawm.latent_dim=8 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=25 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_visual-antmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_visual-antmaze-medium-navigate-v0_${IQL_SEED}/models/latest.pt eval_checkpoint=output/train/telawm_visual-antmaze-medium-navigate-v0_${SEED}/models/latest.pt seed=${SEED}

#### Visual AntMaze Medium Stitch ####

## AE ##
python train.py steps=200000 task=visual-antmaze-medium-stitch-v0 obs=rgb batch_size=256 eval_freq=25000 agent=ae seed=${SEED}

## IQL ##
python train.py steps=501000 task=visual-antmaze-medium-stitch-v0 obs=rgb batch_size=256 gamma=0.99 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 eval_freq=250000 agent=iql iql.expectile=0.7 iql.alpha=0.1 seed=${SEED}

## HIQL ##
python train.py steps=1001000 task=visual-antmaze-medium-stitch-v0 obs=rgb batch_size=256 gamma=0.99 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 eval_freq=250000 agent=hiql latent_encoder_checkpoint=output/train/ae_visual-antmaze-medium-stitch-v0_${AE_SEED}/models/latest.pt hiql.beta=3.0 hiql.subgoal_k=25 hiql.subgoal_steps=10 hiql.subgoal_reached_threshold=10 hiql.goal_vicinity_threshold=30 hiql.pretrained_vf=true hiql.vf_chkpt=output/train/iql_visual-antmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt hiql.pretrained_policy=true hiql.policy_chkpt=output/train/iql_visual-antmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SD ##
python train.py steps=1001000 task=visual-antmaze-medium-stitch-v0 obs=latent latent_encoder_checkpoint=output/train/ae_visual-antmaze-medium-stitch-v0_${AE_SEED}/models/latest.pt gamma=0.99 p_trajgoal_policy=0.5 p_randgoal_policy=0.5 agent=sd sd.goal_conditioned=false sd.n_diffusion_steps=20 sd.ddim_sampling_steps=10 sd.sampling_temperature=1.0 sd.subgoal_k=25 sd.n_sg_samples=128 sd.subgoal_steps=10 sd.subgoal_reached_threshold=10 sd.goal_vicinity_threshold=30 sd.pretrained_vf=false sd.pretrained_policy=false seed=${SEED}
python evaluate.py task=visual-antmaze-medium-stitch-v0 obs=rgb latent_encoder_checkpoint=output/train/ae_visual-antmaze-medium-stitch-v0_${AE_SEED}/models/latest.pt gamma=0.99 agent=sd sd.goal_conditioned=false sd.n_diffusion_steps=20 sd.ddim_sampling_steps=10 sd.sampling_temperature=1.0 sd.subgoal_k=25 sd.n_sg_samples=128 sd.subgoal_steps=10 sd.subgoal_reached_threshold=10 sd.goal_vicinity_threshold=30 sd.pretrained_vf=true sd.vf_chkpt=output/train/iql_visual-antmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt sd.pretrained_policy=true sd.policy_chkpt=output/train/iql_visual-antmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt eval_checkpoint=output/train/sd_visual-antmaze-medium-stitch-v0_${SEED}/models/latest.pt seed=${SEED}

## Te-LAWM ##
python train.py steps=1001000 task=visual-antmaze-medium-stitch-v0 obs=latent latent_encoder_checkpoint=output/train/ae_visual-antmaze-medium-stitch-v0_${AE_SEED}/models/latest.pt gamma=0.99 agent=telawm telawm.latent_dim=8 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=25 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=30 seed=${SEED}
python evaluate.py task=visual-antmaze-medium-stitch-v0 obs=rgb latent_encoder_checkpoint=output/train/ae_visual-antmaze-medium-stitch-v0_${AE_SEED}/models/latest.pt gamma=0.99 agent=telawm telawm.latent_dim=8 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=25 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_visual-antmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_visual-antmaze-medium-stitch-v0_${IQL_SEED}/models/latest.pt eval_checkpoint=output/train/telawm_visual-antmaze-medium-stitch-v0_${SEED}/models/latest.pt seed=${SEED}

#### Visual AntMaze Large Navigate ####

## AE ##
python train.py steps=200000 task=visual-antmaze-large-navigate-v0 obs=rgb batch_size=256 eval_freq=25000 agent=ae seed=${SEED}

## IQL ##
python train.py steps=501000 task=visual-antmaze-large-navigate-v0 obs=rgb batch_size=256 gamma=0.999 eval_freq=250000 agent=iql iql.expectile=0.7 iql.alpha=0.1 seed=${SEED}

## HIQL ##
python train.py steps=1001000 task=visual-antmaze-large-navigate-v0 obs=rgb batch_size=256 gamma=0.999 eval_freq=250000 agent=hiql latent_encoder_checkpoint=output/train/ae_visual-antmaze-large-navigate-v0_${AE_SEED}/models/latest.pt hiql.beta=3.0 hiql.subgoal_k=25 hiql.subgoal_steps=10 hiql.subgoal_reached_threshold=10 hiql.goal_vicinity_threshold=30 hiql.pretrained_vf=true hiql.vf_chkpt=output/train/iql_visual-antmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt hiql.pretrained_policy=true hiql.policy_chkpt=output/train/iql_visual-antmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt seed=${SEED}

## SD ##
python train.py steps=1001000 task=visual-antmaze-large-navigate-v0 obs=latent latent_encoder_checkpoint=output/train/ae_visual-antmaze-large-navigate-v0_${AE_SEED}/models/latest.pt gamma=0.999 agent=sd sd.goal_conditioned=false sd.n_diffusion_steps=20 sd.ddim_sampling_steps=10 sd.sampling_temperature=1.0 sd.subgoal_k=50 sd.n_sg_samples=128 sd.subgoal_steps=10 sd.subgoal_reached_threshold=10 sd.goal_vicinity_threshold=30 sd.pretrained_vf=false sd.pretrained_policy=false seed=${SEED}
python evaluate.py task=visual-antmaze-large-navigate-v0 obs=rgb latent_encoder_checkpoint=output/train/ae_visual-antmaze-large-navigate-v0_${AE_SEED}/models/latest.pt gamma=0.999 agent=sd sd.goal_conditioned=false sd.n_diffusion_steps=20 sd.ddim_sampling_steps=10 sd.sampling_temperature=1.0 sd.subgoal_k=50 sd.n_sg_samples=128 sd.subgoal_steps=10 sd.subgoal_reached_threshold=10 sd.goal_vicinity_threshold=30 sd.pretrained_vf=true sd.vf_chkpt=output/train/iql_visual-antmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt sd.pretrained_policy=true sd.policy_chkpt=output/train/iql_visual-antmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt eval_checkpoint=output/train/sd_visual-antmaze-large-navigate-v0_${SEED}/models/latest.pt seed=${SEED}

## Te-LAWM ##
python train.py steps=1001000 task=visual-antmaze-large-navigate-v0 obs=latent latent_encoder_checkpoint=output/train/ae_visual-antmaze-large-navigate-v0_${AE_SEED}/models/latest.pt gamma=0.999 agent=telawm telawm.latent_dim=8 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=50 telawm.subgoal_steps=10 telawm.planning_steps=2 telawm.subgoal_state=first telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=30 seed=${SEED}
python evaluate.py task=visual-antmaze-large-navigate-v0 obs=rgb latent_encoder_checkpoint=output/train/ae_visual-antmaze-large-navigate-v0_${AE_SEED}/models/latest.pt gamma=0.999 agent=telawm telawm.latent_dim=8 telawm.learned_prior=true telawm.beta_kl=0.004 telawm.lam_timescale=50 telawm.subgoal_steps=10 telawm.planning_steps=4 telawm.subgoal_state=final telawm.n_sg_samples=128 telawm.subgoal_reached_threshold=10 telawm.goal_vicinity_threshold=30 telawm.pretrained_vf=true telawm.vf_chkpt=output/train/iql_visual-antmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt telawm.pretrained_policy=true telawm.policy_chkpt=output/train/iql_visual-antmaze-large-navigate-v0_${IQL_SEED}/models/latest.pt eval_checkpoint=output/train/telawm_visual-antmaze-large-navigate-v0_${SEED}/models/latest.pt seed=${SEED}
