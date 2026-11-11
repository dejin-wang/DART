import numpy as np
import torch

import global_context
from garage import TrajectoryBatch
from garagei import log_performance_ex
from iod import sac_utils
from iod.iod import IOD
import copy

from iod.utils import get_torch_concat_obs, FigManager, get_option_colors, record_video, draw_2d_gaussians

# ray_beta_mode 'adaptive': beta(psi) is stored on a periodic grid of skill angles (0.1 deg apart).
RAY_ADAPT_GRID = 3600


class METRA(IOD):
    def __init__(
            self,
            *,
            qf1,
            qf2,
            log_alpha,
            tau,
            scale_reward,
            target_coef,

            replay_buffer,
            min_buffer_size,
            inner,
            num_alt_samples,
            split_group,

            dual_reg,
            dual_slack,
            dual_dist,

            reward_mode='metra',
            ray_beta=0.0,
            ray_warmup=(0, 0),
            ray_beta_mode='fixed',
            ray_adapt_window=1,
            ray_adapt_eta=1.,
            nm_sigma=10.0,
            nm_step=1.0,
            d3_switch_range=(0.5, 0.7),

            pixel_shape=None,

            **kwargs,
    ):
        super().__init__(**kwargs)

        assert reward_mode in ['metra', 'ray', 'norm_match', 'd3']
        if reward_mode in ['ray', 'norm_match', 'd3']:
            assert inner, f'{reward_mode} reward needs the inner-product (METRA/LSD) reward'
        if reward_mode in ['ray', 'norm_match', 'd3']:
            # norm_match / d3 rescale the unit skills for training (see _get_train_trajectories_kwargs); evaluation
            # keeps unit skills, i.e. the largest (fastest) skill.
            assert not self.discrete and self.unit_length, f'{reward_mode} needs continuous unit-length skills'
        self.reward_mode = reward_mode
        self.ray_beta = ray_beta
        self.ray_warmup = ray_warmup  # (start, end) epochs: beta = 0 before start, ramps linearly to ray_beta at end
        # ray_beta_mode 'adaptive' (DART's direction-adaptive lateral penalty, 2-D skills): at the end of every epoch,
        # beta(psi) = ray_beta * (1 + e(psi)) / (1 + mean_psi e), where e(psi) linearly interpolates (periodically in
        # the skill angle psi) the errors e_i = (1 - cos angle(u_T, z_i)) / 2 of the training trajectories of the last
        # ray_adapt_window epochs (1 = the latest epoch only), re-encoded with the current phi. Mean beta stays ray_beta
        # and beta(psi) stays in [ray_beta / 2, 2 ray_beta]; the warmup ramp multiplies it as in the fixed mode.
        assert ray_beta_mode in ['fixed', 'adaptive']
        self.ray_beta_mode = ray_beta_mode
        if ray_beta_mode == 'adaptive':
            assert reward_mode == 'ray' and self.dim_option == 2, 'ray_beta_mode adaptive needs reward_mode ray, 2-D z'
            self._init_ray_adaptive(ray_adapt_window)
            # ray_adapt_eta: beta(psi) = ray_beta (1 + e(psi))^eta / mean_psi (1 + e)^eta (eta = 1: the rule above).
            self.ray_adapt_eta = float(ray_adapt_eta)
        # norm_match / d3: skill z asks for dphi = nm_step * z per step; reward 1 / (1 + nm_sigma * |dphi / nm_step - z|^2).
        self.nm_sigma = nm_sigma
        self.nm_step = nm_step
        self.d3_switch_range = d3_switch_range
        self.d3_alpha = 0.  # d3: weight of norm matching (0 = METRA alignment only), set from the cosine similarity
        self._d3_cos = []
        self._epoch = 0

        self.qf1 = qf1.to(self.device)
        self.qf2 = qf2.to(self.device)

        self.target_qf1 = copy.deepcopy(self.qf1)
        self.target_qf2 = copy.deepcopy(self.qf2)

        self.log_alpha = log_alpha.to(self.device)

        self.param_modules.update(
            qf1=self.qf1,
            qf2=self.qf2,
            log_alpha=self.log_alpha,
        )

        self.tau = tau

        self.replay_buffer = replay_buffer
        self.min_buffer_size = min_buffer_size
        self.inner = inner

        self.dual_reg = dual_reg
        self.dual_slack = dual_slack
        self.dual_dist = dual_dist

        self.num_alt_samples = num_alt_samples
        self.split_group = split_group

        self._reward_scale_factor = scale_reward
        self._target_entropy = -np.prod(self._env_spec.action_space.shape).item() / 2. * target_coef

        self.pixel_shape = pixel_shape

        assert self._trans_optimization_epochs is not None

    @property
    def policy(self):
        return {
            'option_policy': self.option_policy,
        }

    def _get_concat_obs(self, obs, option):
        return get_torch_concat_obs(obs, option)

    def _get_policy_obs(self, v):
        """Policy/Q-function input (for s and s') before the skill is appended: s, or (s, u) in the ray mode."""
        if self.reward_mode == 'ray':
            return torch.cat([v['obs'], v['ray_u']], dim=1), torch.cat([v['next_obs'], v['ray_next_u']], dim=1)
        return v['obs'], v['next_obs']

    def _ray_beta_now(self, z=None):
        """beta for skills z: ray_beta (fixed) or beta(z) (adaptive), times the warmup ramp."""
        start, end = self.ray_warmup
        if end <= start:
            ramp = float(self._epoch >= start)
        else:
            ramp = float(np.clip((self._epoch - start) / (end - start), 0., 1.))
        if self.ray_beta_mode == 'adaptive' and z is not None:
            return ramp * self._ray_beta_of(z)
        return self.ray_beta * ramp

    def _ray_beta_of(self, z):
        """Adaptive ray mode: beta(z) linearly interpolated on the angle grid."""
        x = torch.remainder(torch.atan2(z[:, 1], z[:, 0]), 2 * np.pi) * (RAY_ADAPT_GRID / (2 * np.pi))
        i = x.floor().long().clamp(0, RAY_ADAPT_GRID - 1)
        frac = x - i
        return (1 - frac) * self._ray_beta_grid[i] + frac * self._ray_beta_grid[(i + 1) % RAY_ADAPT_GRID]

    def _init_ray_adaptive(self, window):
        self.ray_adapt_window = int(window)
        self._ray_beta_grid = torch.full((RAY_ADAPT_GRID,), float(self.ray_beta), device=self.device)
        self._ray_adapt_epochs = []  # per epoch: (z, s_0, s_T) of its training trajectories

    def _update_ray_adaptive(self, path_data):
        """Adaptive ray mode: recompute beta(psi) from the end-angle errors of the recent training trajectories."""
        self._ray_adapt_epochs.append([(options[0], obs[0], next_obs[-1]) for obs, next_obs, options
                                       in zip(path_data['obs'], path_data['next_obs'], path_data['options'])])
        self._ray_adapt_epochs = self._ray_adapt_epochs[-self.ray_adapt_window:]
        trajs = [t for epoch in self._ray_adapt_epochs for t in epoch]
        z, s0, sT = (torch.as_tensor(np.stack(x), dtype=torch.float32, device=self.device) for x in zip(*trajs))
        with torch.no_grad():
            u = self.traj_encoder(sT).mean - self.traj_encoder(s0).mean
            norm = u.norm(dim=1)
            valid = norm > 1e-8
            cos = (u * z).sum(dim=1)[valid] / norm[valid]
            err = (0.5 * (1 - cos)).cpu().numpy()  # sin^2(angle / 2) in [0, 1]
            psi = np.mod(np.arctan2(z[valid, 1].cpu().numpy(), z[valid, 0].cpu().numpy()), 2 * np.pi)
            if len(err):  # no valid displacement: beta unchanged
                grid = 2 * np.pi * np.arange(RAY_ADAPT_GRID) / RAY_ADAPT_GRID
                order = np.argsort(psi)
                e_hat = np.interp(grid, psi[order], err[order], period=2 * np.pi)
                w = (1 + e_hat) ** getattr(self, 'ray_adapt_eta', 1.)
                beta = self.ray_beta * w / w.mean()
                self._ray_beta_grid = torch.as_tensor(beta, dtype=torch.float32, device=self.device)
            beta = self._ray_beta_grid
            angle = torch.rad2deg(torch.acos(cos.clamp(-1, 1))) if len(err) else torch.zeros(1)
        return {
            'RayAdaptTrajs': torch.tensor(float(len(err))),
            'RayEndAngleMean': angle.mean(),
            'RayEndAngleMax': angle.max(),
            'RayAdaptErrMean': torch.tensor(float(err.mean()) if len(err) else 0.),
            'RayBetaGridMin': beta.min(),
            'RayBetaGridMean': beta.mean(),
            'RayBetaGridMax': beta.max(),
            'RayBetaAnchors': beta[::RAY_ADAPT_GRID // 36],  # beta every 10 deg, from 0 deg
        }

    def train_once(self, itr, paths, runner, extra_scalar_metrics={}):
        self._epoch = itr  # read by the ray beta warmup
        return super().train_once(itr, paths, runner, extra_scalar_metrics)

    def process_samples(self, paths):
        data = super().process_samples(paths)
        if self.reward_mode == 'ray':
            # Each transition keeps its trajectory's s_0, so u_t = phi(s_t) - phi(s_0) can be recomputed with the current phi.
            data['start_obs'] = [np.repeat(path['observations'][:1], len(path['observations']), axis=0) for path in paths]
        return data

    def _get_policy_param_values(self, key):
        if self.reward_mode == 'ray':
            # Rollouts compute u_t with the policy's own copy of phi; bring it up to date first.
            self.option_policy.ray_phi.load_state_dict(self.traj_encoder.state_dict())
        return super()._get_policy_param_values(key)

    def _get_train_trajectories_kwargs(self, runner):
        if self.discrete:
            extras = self._generate_option_extras(np.eye(self.dim_option)[np.random.randint(0, self.dim_option, runner._train_args.batch_size)])
        else:
            random_options = np.random.randn(runner._train_args.batch_size, self.dim_option)
            if self.unit_length:
                random_options /= np.linalg.norm(random_options, axis=-1, keepdims=True)
            n = len(random_options)
            if self.reward_mode == 'norm_match':
                # Atanassov et al.: z uniform in the disk (ball), so |z| sets the speed.
                random_options *= np.random.rand(n, 1) ** (1 / self.dim_option)
            elif self.reward_mode == 'd3':
                # D3's code: with probability alpha a norm-matching skill of radius U(0, 1.5), otherwise a unit skill.
                radii = np.where(np.random.rand(n, 1) < self.d3_alpha, np.random.rand(n, 1) * 1.5, 1.)
                random_options *= radii
            extras = self._generate_option_extras(random_options)

        return dict(
            extras=extras,
            sampler_key='option_policy',
        )

    def _flatten_data(self, data):
        epoch_data = {}
        for key, value in data.items():
            epoch_data[key] = torch.tensor(np.concatenate(value, axis=0), dtype=torch.float32, device=self.device)
        return epoch_data

    def _update_replay_buffer(self, data):
        if self.replay_buffer is not None:
            # Add paths to the replay buffer
            for i in range(len(data['actions'])):
                path = {}
                for key in data.keys():
                    cur_list = data[key][i]
                    if cur_list.ndim == 1:
                        cur_list = cur_list[..., np.newaxis]
                    path[key] = cur_list
                self.replay_buffer.add_path(path)

    def _sample_replay_buffer(self):
        samples = self.replay_buffer.sample_transitions(self._trans_minibatch_size)
        data = {}
        for key, value in samples.items():
            if value.shape[1] == 1 and 'option' not in key:
                value = np.squeeze(value, axis=1)
            data[key] = torch.from_numpy(value).float().to(self.device)
        return data

    def _train_once_inner(self, path_data):
        self._update_replay_buffer(path_data)

        epoch_data = self._flatten_data(path_data)
        if self.reward_mode == 'd3':
            self._update_d3_alpha(epoch_data)

        tensors = self._train_components(epoch_data)
        if self.ray_beta_mode == 'adaptive':
            adapt_stats = self._update_ray_adaptive(path_data)
            if tensors:  # (not before SAC starts: the first logged row fixes the progress.csv columns)
                tensors.update(adapt_stats)

        return tensors

    def _update_d3_alpha(self, epoch_data):
        """D3's switch from METRA to norm matching: alpha ramps 0 -> 1 as the mean cosine similarity between dphi and z
        on fresh rollouts (here the last 4 epochs) goes from d3_switch_range[0] to [1]; it can also move back."""
        with torch.no_grad():
            dphi = self.traj_encoder(epoch_data['next_obs']).mean - self.traj_encoder(epoch_data['obs']).mean
            cos = torch.nn.functional.cosine_similarity(dphi, epoch_data['options'], dim=1).mean().item()
        self._d3_cos = (self._d3_cos + [cos])[-4:]
        lo, hi = self.d3_switch_range
        self.d3_alpha = float(np.clip((np.mean(self._d3_cos) - lo) / (hi - lo), 0., 1.))

    def _train_components(self, epoch_data):
        if self.replay_buffer is not None and self.replay_buffer.n_transitions_stored < self.min_buffer_size:
            return {}

        for _ in range(self._trans_optimization_epochs):
            tensors = {}

            if self.replay_buffer is None:
                v = self._get_mini_tensors(epoch_data)
            else:
                v = self._sample_replay_buffer()

            self._optimize_te(tensors, v)
            self._update_rewards(tensors, v)
            self._optimize_op(tensors, v)

        return tensors

    def _optimize_te(self, tensors, internal_vars):
        self._update_loss_te(tensors, internal_vars)

        self._gradient_descent(
            tensors['LossTe'],
            optimizer_keys=['traj_encoder'],
        )

        if self.dual_reg:
            self._update_loss_dual_lam(tensors, internal_vars)
            self._gradient_descent(
                tensors['LossDualLam'],
                optimizer_keys=['dual_lam'],
            )
            if self.dual_dist == 's2_from_s':
                self._gradient_descent(
                    tensors['LossDp'],
                    optimizer_keys=['dist_predictor'],
                )

    def _optimize_op(self, tensors, internal_vars):
        self._update_loss_qf(tensors, internal_vars)

        self._gradient_descent(
            tensors['LossQf1'] + tensors['LossQf2'],
            optimizer_keys=['qf'],
        )

        self._update_loss_op(tensors, internal_vars)
        self._gradient_descent(
            tensors['LossSacp'],
            optimizer_keys=['option_policy'],
        )

        self._update_loss_alpha(tensors, internal_vars)
        self._gradient_descent(
            tensors['LossAlpha'],
            optimizer_keys=['log_alpha'],
        )

        sac_utils.update_targets(self)

    def _update_rewards(self, tensors, v):
        obs = v['obs']
        next_obs = v['next_obs']

        if self.inner:
            cur_z = self.traj_encoder(obs).mean
            next_z = self.traj_encoder(next_obs).mean
            target_z = next_z - cur_z

            if self.discrete:
                masks = (v['options'] - v['options'].mean(dim=1, keepdim=True)) * self.dim_option / (self.dim_option - 1 if self.dim_option != 1 else 1)
                rewards = (target_z * masks).sum(dim=1)
            else:
                inner = (target_z * v['options']).sum(dim=1)
                rewards = inner

            if self.reward_mode == 'ray':
                # SAC reward z^T dphi - beta * (lat(u') - lat(u)), u = phi(s) - phi(s_0), lat(u) = |u - (z^T u) z|.
                # Summed over a trajectory: z^T u_T - beta * lat(u_T). The phi objective keeps using `rewards`.
                with torch.no_grad():
                    z = v['options']
                    phi_start = self.traj_encoder(v['start_obs']).mean
                    u, next_u = cur_z - phi_start, next_z - phi_start
                    lat = (u - (u * z).sum(dim=1, keepdim=True) * z).norm(dim=1)
                    next_lat = (next_u - (next_u * z).sum(dim=1, keepdim=True) * z).norm(dim=1)
                    beta = self._ray_beta_now(z)
                    v['policy_rewards'] = rewards - beta * (next_lat - lat)
                v.update({
                    'ray_u': u,
                    'ray_next_u': next_u,
                })
                tensors.update({
                    'RayBeta': beta.mean() if torch.is_tensor(beta) else torch.tensor(beta),
                    'RayLatDeltaMean': (next_lat - lat).mean(),
                    'RayLatMean': next_lat.mean(),
                    'RayAdvMean': (next_u * z).sum(dim=1).mean(),
                    'PolicyRewardMean': v['policy_rewards'].mean(),
                })
            elif self.reward_mode in ['norm_match', 'd3']:
                # The phi objective uses v['te_rewards'] instead of the alignment `rewards`.
                z = v['options']
                rel = target_z / self.nm_step  # dphi in units of the skill: matched when rel = z
                with torch.no_grad():
                    nm_err = ((rel - z) ** 2).sum(dim=1)
                    nm_rewards = 1 / (1 + self.nm_sigma * nm_err)
                if self.reward_mode == 'norm_match':
                    # Atanassov et al.: N dphi should equal R z (R = N * nm_step); Smooth L1 phi loss with beta = 0.1 R
                    # (their 5 for R = 50); reward 1 / (1 + sigma |N dphi - R z|^2 / R^2) = nm_rewards.
                    N = self.max_path_length
                    R = N * self.nm_step
                    v['te_rewards'] = -torch.nn.functional.smooth_l1_loss(
                        N * target_z, R * z, beta=0.1 * R, reduction='none').mean(dim=1)
                    v['policy_rewards'] = nm_rewards
                else:
                    # D3 (Cathomen et al., released code): both rewards z-scored over the batch, then mixed with alpha;
                    # the phi loss mixes -alignment and Smooth L1 (beta 1, summed) matching. In D3's units the constraint
                    # is |dphi| <= 1; here it is |dphi| <= sqrt(dim_option), so nm_step = sqrt(dim_option) and the
                    # matching term is scaled by nm_step to keep D3's ratio between the two terms.
                    a = self.d3_alpha
                    zscore = lambda r: (r - r.mean()) / (r.std() + 1e-6)
                    v['te_rewards'] = (1 - a) * rewards - a * self.nm_step * torch.nn.functional.smooth_l1_loss(
                        rel, z, reduction='none').sum(dim=1)
                    with torch.no_grad():
                        v['policy_rewards'] = (1 - a) * zscore(rewards) + a * zscore(nm_rewards)
                    tensors.update({
                        'D3Alpha': torch.tensor(a),
                        'D3CosSim': torch.tensor(float(np.mean(self._d3_cos)) if self._d3_cos else 0.),
                    })
                tensors.update({
                    'NmRewardMean': nm_rewards.mean(),
                    'NmErrMean': nm_err.sqrt().mean(),
                    'PolicyRewardMean': v['policy_rewards'].mean(),
                    'PhiStepNormMean': target_z.norm(dim=1).mean(),
                    'CosSimMean': torch.nn.functional.cosine_similarity(target_z, z, dim=1).mean(),
                })

            # For dual objectives
            v.update({
                'cur_z': cur_z,
                'next_z': next_z,
            })
        else:
            target_dists = self.traj_encoder(next_obs)

            if self.discrete:
                logits = target_dists.mean
                rewards = -torch.nn.functional.cross_entropy(logits, v['options'].argmax(dim=1), reduction='none')
            else:
                rewards = target_dists.log_prob(v['options'])

        tensors.update({
            'PureRewardMean': rewards.mean(),
            'PureRewardStd': rewards.std(),
        })

        v['rewards'] = rewards

    def _update_loss_te(self, tensors, v):
        self._update_rewards(tensors, v)
        rewards = v.get('te_rewards', v['rewards'])

        obs = v['obs']
        next_obs = v['next_obs']

        if self.dual_dist == 's2_from_s':
            s2_dist = self.dist_predictor(obs)
            loss_dp = -s2_dist.log_prob(next_obs - obs).mean()
            tensors.update({
                'LossDp': loss_dp,
            })

        if self.dual_reg:
            dual_lam = self.dual_lam.param.exp()
            x = obs
            y = next_obs
            phi_x = v['cur_z']
            phi_y = v['next_z']

            if self.dual_dist == 'l2':
                cst_dist = torch.square(y - x).mean(dim=1)
            elif self.dual_dist == 'one':
                cst_dist = torch.ones_like(x[:, 0])
            elif self.dual_dist == 's2_from_s':
                s2_dist = self.dist_predictor(obs)
                s2_dist_mean = s2_dist.mean
                s2_dist_std = s2_dist.stddev
                scaling_factor = 1. / s2_dist_std
                geo_mean = torch.exp(torch.log(scaling_factor).mean(dim=1, keepdim=True))
                normalized_scaling_factor = (scaling_factor / geo_mean) ** 2
                cst_dist = torch.mean(torch.square((y - x) - s2_dist_mean) * normalized_scaling_factor, dim=1)

                tensors.update({
                    'ScalingFactor': scaling_factor.mean(dim=0),
                    'NormalizedScalingFactor': normalized_scaling_factor.mean(dim=0),
                })
            else:
                raise NotImplementedError

            cst_penalty = cst_dist - torch.square(phi_y - phi_x).mean(dim=1)
            cst_penalty = torch.clamp(cst_penalty, max=self.dual_slack)
            te_obj = rewards + dual_lam.detach() * cst_penalty

            v.update({
                'cst_penalty': cst_penalty
            })
            tensors.update({
                'DualCstPenalty': cst_penalty.mean(),
            })
        else:
            te_obj = rewards

        loss_te = -te_obj.mean()

        tensors.update({
            'TeObjMean': te_obj.mean(),
            'LossTe': loss_te,
        })

    def _update_loss_dual_lam(self, tensors, v):
        log_dual_lam = self.dual_lam.param
        dual_lam = log_dual_lam.exp()
        loss_dual_lam = log_dual_lam * (v['cst_penalty'].detach()).mean()

        tensors.update({
            'DualLam': dual_lam,
            'LossDualLam': loss_dual_lam,
        })

    def _update_loss_qf(self, tensors, v):
        obs, next_obs = self._get_policy_obs(v)
        processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(obs), v['options'])
        next_processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(next_obs), v['next_options'])

        sac_utils.update_loss_qf(
            self, tensors, v,
            obs=processed_cat_obs,
            actions=v['actions'],
            next_obs=next_processed_cat_obs,
            dones=v['dones'],
            rewards=v.get('policy_rewards', v['rewards']) * self._reward_scale_factor,
            policy=self.option_policy,
        )

        v.update({
            'processed_cat_obs': processed_cat_obs,
            'next_processed_cat_obs': next_processed_cat_obs,
        })

    def _update_loss_op(self, tensors, v):
        obs, _ = self._get_policy_obs(v)
        processed_cat_obs = self._get_concat_obs(self.option_policy.process_observations(obs), v['options'])
        sac_utils.update_loss_sacp(
            self, tensors, v,
            obs=processed_cat_obs,
            policy=self.option_policy,
        )

    def _update_loss_alpha(self, tensors, v):
        sac_utils.update_loss_alpha(
            self, tensors, v,
        )

    def _evaluate_policy(self, runner):
        if self.discrete:
            eye_options = np.eye(self.dim_option)
            random_options = []
            colors = []
            for i in range(self.dim_option):
                num_trajs_per_option = self.num_random_trajectories // self.dim_option + (i < self.num_random_trajectories % self.dim_option)
                for _ in range(num_trajs_per_option):
                    random_options.append(eye_options[i])
                    colors.append(i)
            random_options = np.array(random_options)
            colors = np.array(colors)
            num_evals = len(random_options)
            from matplotlib import cm
            cmap = 'tab10' if self.dim_option <= 10 else 'tab20'
            random_option_colors = []
            for i in range(num_evals):
                random_option_colors.extend([cm.get_cmap(cmap)(colors[i])[:3]])
            random_option_colors = np.array(random_option_colors)
        else:
            random_options = np.random.randn(self.num_random_trajectories, self.dim_option)
            if self.unit_length:
                random_options = random_options / np.linalg.norm(random_options, axis=1, keepdims=True)
            random_option_colors = get_option_colors(random_options * 4)
        random_trajectories = self._get_trajectories(
            runner,
            sampler_key='option_policy',
            extras=self._generate_option_extras(random_options),
            worker_update=dict(
                _render=False,
                _deterministic_policy=True,
            ),
            env_update=dict(_action_noise_std=None),
        )

        with FigManager(runner, 'TrajPlot_RandomZ') as fm:
            runner._env.render_trajectories(
                random_trajectories, random_option_colors, self.eval_plot_axis, fm.ax
            )

        data = self.process_samples(random_trajectories)
        last_obs = torch.stack([torch.from_numpy(ob[-1]).to(self.device) for ob in data['obs']])
        option_dists = self.traj_encoder(last_obs)

        option_means = option_dists.mean.detach().cpu().numpy()
        if self.inner:
            option_stddevs = torch.ones_like(option_dists.stddev.detach().cpu()).numpy()
        else:
            option_stddevs = option_dists.stddev.detach().cpu().numpy()
        option_samples = option_dists.mean.detach().cpu().numpy()

        option_colors = random_option_colors

        with FigManager(runner, f'PhiPlot') as fm:
            draw_2d_gaussians(option_means, option_stddevs, option_colors, fm.ax)
            draw_2d_gaussians(
                option_samples,
                [[0.03, 0.03]] * len(option_samples),
                option_colors,
                fm.ax,
                fill=True,
                use_adaptive_axis=True,
            )

        eval_option_metrics = {}

        # Videos
        if self.eval_record_video:
            if self.discrete:
                video_options = np.eye(self.dim_option)
                video_options = video_options.repeat(self.num_video_repeats, axis=0)
            else:
                if self.dim_option == 2:
                    radius = 1. if self.unit_length else 1.5
                    video_options = []
                    for angle in [3, 2, 1, 4]:
                        video_options.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                    video_options.append([0, 0])
                    for angle in [0, 5, 6, 7]:
                        video_options.append([radius * np.cos(angle * np.pi / 4), radius * np.sin(angle * np.pi / 4)])
                    video_options = np.array(video_options)
                else:
                    video_options = np.random.randn(9, self.dim_option)
                    if self.unit_length:
                        video_options = video_options / np.linalg.norm(video_options, axis=1, keepdims=True)
                video_options = video_options.repeat(self.num_video_repeats, axis=0)
            video_trajectories = self._get_trajectories(
                runner,
                sampler_key='local_option_policy',
                extras=self._generate_option_extras(video_options),
                worker_update=dict(
                    _render=True,
                    _deterministic_policy=True,
                ),
            )
            record_video(runner, 'Video_RandomZ', video_trajectories, skip_frames=self.video_skip_frames)

        eval_option_metrics.update(runner._env.calc_eval_metrics(random_trajectories, is_option_trajectories=True))
        if self.reward_mode == 'ray':
            eval_option_metrics.update(self._ray_eval_metrics(random_trajectories, random_options))
        with global_context.GlobalContext({'phase': 'eval', 'policy': 'option'}):
            log_performance_ex(
                runner.step_itr,
                TrajectoryBatch.from_trajectory_list(self._env_spec, random_trajectories),
                discount=self.discount,
                additional_records=eval_option_metrics,
            )
        self._log_eval_metrics(runner)

    def _ray_eval_metrics(self, trajectories, options):
        """End-of-skill progress along (adv) and off (lat) the ray z, and phi units per metre walked."""
        adv, lat, phi_len, xy_len = [], [], 0., 0.
        for traj, z in zip(trajectories, options):
            obs = np.concatenate([traj['observations'], traj['next_observations'][-1:]])
            with torch.no_grad():
                phi = self.traj_encoder(torch.as_tensor(obs, dtype=torch.float32, device=self.device)).mean.cpu().numpy()
            u = phi[-1] - phi[0]
            adv.append(u @ z)
            lat.append(np.linalg.norm(u - (u @ z) * z))
            if 'coordinates' in traj['env_infos']:
                xy = np.concatenate([traj['env_infos']['coordinates'], traj['env_infos']['next_coordinates'][-1:]])
                phi_len += np.linalg.norm(np.diff(phi, axis=0), axis=1).sum()
                xy_len += np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
        angle = np.degrees(np.arctan2(lat, adv))
        metrics = {
            'RayEndAdvMean': float(np.mean(adv)),
            'RayEndLatMean': float(np.mean(lat)),
            'RayEndAngleMeanDeg': float(np.mean(angle)),
            'RayEndAngleP90Deg': float(np.percentile(angle, 90)),
            'RayEndAngleMaxDeg': float(np.max(angle)),
        }
        if xy_len > 0:
            metrics['PhiUnitsPerMetre'] = float(phi_len / xy_len)
        return metrics
