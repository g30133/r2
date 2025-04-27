from steve1.data.text_alignment.vae import load_vae_model
from steve1.utils.mineclip_agent_env_utils import load_mineclip_agent_env
from steve1.config import PRIOR_INFO, DEVICE
from steve1.utils.embed_utils import get_prior_embed

import torch
import time

prompt = "dig"

def init(in_model, in_weights, prior_info, cond_scale):
    print('====init====')
    agent, mineclip, env = load_mineclip_agent_env(in_model, in_weights, None, cond_scale)
    obs = env.reset()
    prior = load_vae_model(prior_info)
    prompt_embed = get_prior_embed(prompt, mineclip, prior, DEVICE)
    print('====init DONE====')
    return agent, env, obs, prompt_embed


agent, env, obs, prompt_embed = init("data/weights/vpt/2x.model", "data/weights/steve1/steve1.weights", PRIOR_INFO, 6.0)


with torch.cuda.amp.autocast(): minerl_action, minerl_action0, action_mapper, agent_action, hidden_state, result = agent.get_action(obs, prompt_embed)




obs, _, _, _ = env.step(action)


