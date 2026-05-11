"""Public package entry points for adversarial_lassonet."""

__all__ = [
    "AdversarialLassoNetClassifier",
    "LassoNetSAMAligned",
    "build_adv_arg_parser",
    "canonical_dataset_name",
    "namespace_to_adversarial_lassonet_config",
    "resolve_datasets",
    "resolve_device",
    "save_sparse_checkpoint",
    "set_seed",
    "train_adversarial_lassonet_once",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from models.adversarial_lassonet import (
        AdversarialLassoNetClassifier,
        LassoNetSAMAligned,
        build_adv_arg_parser,
        canonical_dataset_name,
        namespace_to_adversarial_lassonet_config,
        resolve_datasets,
        resolve_device,
        save_sparse_checkpoint,
        set_seed,
        train_adversarial_lassonet_once,
    )

    exports = {
        "AdversarialLassoNetClassifier": AdversarialLassoNetClassifier,
        "LassoNetSAMAligned": LassoNetSAMAligned,
        "build_adv_arg_parser": build_adv_arg_parser,
        "canonical_dataset_name": canonical_dataset_name,
        "namespace_to_adversarial_lassonet_config": namespace_to_adversarial_lassonet_config,
        "resolve_datasets": resolve_datasets,
        "resolve_device": resolve_device,
        "save_sparse_checkpoint": save_sparse_checkpoint,
        "set_seed": set_seed,
        "train_adversarial_lassonet_once": train_adversarial_lassonet_once,
    }
    value = exports[name]
    globals()[name] = value
    return value
