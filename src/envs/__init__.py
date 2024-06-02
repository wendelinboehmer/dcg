from functools import partial
from smacv2.env import MultiAgentEnv, StarCraft2Env, StarCraftCapabilityEnvWrapper
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["sc2wrapped"] = partial(env_fn, env=StarCraftCapabilityEnvWrapper)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))

from .stag_hunt import StagHunt
REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)
