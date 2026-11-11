import torch

from garagei.torch.policies.policy_ex import PolicyEx


class RayPolicyEx(PolicyEx):
    """Policy for reward_mode='ray': the network input is (s, u_t, z) with u_t = phi(s_t) - phi(s_0).

    forward() takes that input as is (training, where METRA builds u_t from the current phi).
    get_actions() takes (s, z) like PolicyEx and inserts u_t itself (rollouts), using its own copy of phi
    (`ray_phi`, synced by METRA before every rollout) and the phi(s_0) remembered since the last reset().
    """

    def __init__(self, name, *, ray_phi, obs_dim, **kwargs):
        super().__init__(name, **kwargs)
        self.ray_phi = ray_phi.requires_grad_(False)
        self._obs_dim = obs_dim
        self._phi_start = None

    def reset(self, dones=None):
        self._phi_start = None

    def get_actions(self, observations):
        observations = torch.as_tensor(observations).float().to(next(self.parameters()).device)
        obs, option = observations[:, :self._obs_dim], observations[:, self._obs_dim:]
        with torch.no_grad():
            phi = self.ray_phi(obs).mean
        if self._phi_start is None:
            self._phi_start = phi
        return super().get_actions(torch.cat([obs, phi - self._phi_start, option], dim=1))
