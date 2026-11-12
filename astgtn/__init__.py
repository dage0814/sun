from .config import AblationConfig, ModelConfig, TextEncoderConfig, TrainingConfig

__all__ = [
    "AblationConfig",
    "ModelConfig",
    "TextEncoderConfig",
    "TrainingConfig",
    "AdaptiveSpatioTemporalGraphTransformer",
    "MultiModalASTGTN",
]


def __getattr__(name):
    if name == "AdaptiveSpatioTemporalGraphTransformer":
        from .model import AdaptiveSpatioTemporalGraphTransformer

        return AdaptiveSpatioTemporalGraphTransformer
    if name == "MultiModalASTGTN":
        from .pipeline import MultiModalASTGTN

        return MultiModalASTGTN
    raise AttributeError(name)
