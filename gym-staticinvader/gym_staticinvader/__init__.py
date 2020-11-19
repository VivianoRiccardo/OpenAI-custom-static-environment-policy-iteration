from gym.envs.registration import register

register(
    id='staticinvader-v0',
    entry_point='gym_staticinvader.envs:StaticinvaderEnv',
)
register(
    id='staticinvader-extrahard-v0',
    entry_point='gym_staticinvader.envs:StaticinvaderExtraHardEnv',
)
