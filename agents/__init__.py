from agents.iql import IQL
from agents.tdmpc import TDMPC
from agents.saw import SAW
from agents.hiql import HIQL
from agents.sd import SD
from agents.telawm import TeLAWM
from agents.acwm import ACWM
from agents.lacwm import LACWM
from agents.ae import AE


def make_agent(cfg):
    """
	Make an agent for offline training.
	"""
    if cfg.agent == 'iql':
        agent = IQL(cfg)
    elif cfg.agent == 'tdmpc':
        agent = TDMPC(cfg)
    elif cfg.agent == 'saw':
        agent = SAW(cfg)
    elif cfg.agent == 'hiql':
        agent = HIQL(cfg)
    elif cfg.agent == 'sd':
        agent = SD(cfg)
    elif cfg.agent == 'telawm':
        agent = TeLAWM(cfg)
    elif cfg.agent == 'acwm':
        agent = ACWM(cfg)
    elif cfg.agent == 'lacwm':
        agent = LACWM(cfg)
    elif cfg.agent == 'ae':
        agent = AE(cfg)

    return agent
