"""Agent factory functions.

`make_sac_state_agent`  -- low-dim SAC (no encoders); also the residual base case.
`make_sac_pixel_agent`  -- pixel SAC with a ResNet encoder (RL from scratch).
The residual (EXPO-FT) factory lives in `wrl.agents.residual_sac`.
"""

from functools import partial
from typing import Optional

import jax
from jax import nn

from wrl.agents.sac import SACAgent
from wrl.common.typing import Batch, PRNGKey
from wrl.vision.data_augmentations import batched_random_crop


def make_batch_augmentation_func(image_keys) -> callable:
    def data_augmentation_fn(rng, observations):
        for pixel_key in image_keys:
            observations = observations.copy(
                add_or_replace={
                    pixel_key: batched_random_crop(
                        observations[pixel_key], rng, padding=4, num_batch_dims=2
                    )
                }
            )
        return observations

    def augment_batch(batch: Batch, rng: PRNGKey) -> Batch:
        rng, obs_rng, next_obs_rng = jax.random.split(rng, 3)
        obs = data_augmentation_fn(obs_rng, batch["observations"])
        next_obs = data_augmentation_fn(next_obs_rng, batch["next_observations"])
        return batch.copy(
            add_or_replace={"observations": obs, "next_observations": next_obs}
        )

    return augment_batch


def make_sac_pixel_agent(
    seed,
    sample_obs,
    sample_action,
    image_keys=(),
    encoder_type="resnet",
    reward_bias=0.0,
    target_entropy=None,
    discount=0.97,
    critic_ensemble_size=2,
    critic_subsample_size=None,
):
    """Pixel SAC — RL from scratch. `encoder_type` defaults to from-scratch
    `resnet`; pass `resnet-pretrained` to warm-start the encoder."""
    return SACAgent.create_pixels(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs={
            "tanh_squash_distribution": True,
            "std_parameterization": "exp",
            "std_min": 1e-5,
            "std_max": 5,
        },
        critic_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        policy_network_kwargs={
            "activations": nn.tanh,
            "use_layer_norm": True,
            "hidden_dims": [256, 256],
        },
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=critic_ensemble_size,
        critic_subsample_size=critic_subsample_size,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        augmentation_function=make_batch_augmentation_func(image_keys),
    )


def make_sac_state_agent(
    seed,
    sample_obs,
    sample_action,
    *,
    hidden_dims=(256, 256),
    discount=0.99,
    reward_bias=0.0,
    target_entropy=None,
    init_final: Optional[float] = None,
):
    """State-only SAC agent — no image encoders. Use for low-dim gym envs."""
    from wrl.networks.actor_critic_nets import Critic, Policy, ensemblize
    from wrl.networks.lagrange import GeqLagrangeMultiplier
    from wrl.networks.mlp import MLP

    rng = jax.random.PRNGKey(seed)

    critic_backbone = partial(
        MLP,
        hidden_dims=list(hidden_dims),
        activations=nn.tanh,
        use_layer_norm=True,
        activate_final=True,
    )
    critic_backbone = ensemblize(critic_backbone, 2)(name="critic_ensemble")
    critic_def = Critic(encoder=None, network=critic_backbone, name="critic")

    policy_def = Policy(
        encoder=None,
        network=MLP(
            hidden_dims=list(hidden_dims),
            activations=nn.tanh,
            use_layer_norm=True,
            activate_final=True,
        ),
        action_dim=sample_action.shape[-1],
        tanh_squash_distribution=True,
        std_parameterization="exp",
        std_min=1e-5,
        std_max=5.0,
        init_final=init_final,
        name="actor",
    )

    temperature_def = GeqLagrangeMultiplier(
        init_value=1e-2,
        constraint_shape=(),
        constraint_type="geq",
        name="temperature",
    )

    return SACAgent.create(
        rng,
        sample_obs,
        sample_action,
        actor_def=policy_def,
        critic_def=critic_def,
        temperature_def=temperature_def,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=2,
        critic_subsample_size=None,
        reward_bias=reward_bias,
        target_entropy=target_entropy,
        image_keys=None,
        augmentation_function=None,
    )
