from scripts.iros_challenge.onsite_competition.sdk.save_obs import load_obs_from_meta
from internnav.configs.agent import AgentCfg
from internnav.utils import AgentClient

cfg = AgentCfg(
    server_host='localhost',
    server_port=8087,
    model_name='internvla_n1',
    ckpt_path='',
    model_settings={
        'policy_name': "InternVLAN1_Policy",
        'state_encoder': None,
        'env_num': 1,
        'sim_num': 1,
        'model_path': "checkpoints/InternVLA-N1-DualVLN",
        'camera_intrinsic': [[585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]],
        'width': 640,
        'height': 480,
        'hfov': 79,
        'resize_w': 384,
        'resize_h': 384,
        'max_new_tokens': 1024,
        'num_frames': 32,
        'num_history': 8,
        'num_future_steps': 4,
        'device': 'cuda:0',
        'predict_step_nums': 32,
        'continuous_traj': True,
        'vis_debug': False,
    }
)

agent = AgentClient(cfg)

obs = load_obs_from_meta('scripts/iros_challenge/onsite_competition/captures/rs_meta.json')
obs['instruction'] = 'go to the red car'
print(obs['rgb'].shape, obs['rgb'].dtype)
print(obs['depth'].shape, obs['depth'].dtype)
action = agent.step([obs])[0]['action'][0]
print(action)