from training.datasets.from_dataset_manifest import miditok_sources_from_dataset_manifest
from training.datasets.tokenized_dataset import (
    TOKENIZATION_ROLES,
    TokenizedDatasetSummary,
    TokenizedSegment,
    TokenizedSegmentMetadata,
    export_tokenized_dataset,
    load_tokenized_segments,
)
from training.models.role_ngram import (
    CUSTOM_ROLE_NGRAM_MODEL_TYPE,
    CUSTOM_ROLE_NGRAM_VERSION,
    CUSTOM_ROLE_TRAINING_ROLES,
    CustomRoleNgramTrainingSummary,
    RoleNgramCheckpoint,
    RoleNgramModel,
    RoleTrainingSegment,
    checkpoint_dir_for_role,
    load_role_training_segments,
    train_custom_role_ngram_checkpoints,
)
from training.models.statistical import (
    BASELINE_ROLE_MODEL_TYPES,
    StatisticalBaselineSummary,
    StatisticalRoleModel,
    StatisticalRoleModelArtifact,
    train_baseline_statistical_models,
)
from training.tokenizers.miditok_real import (
    MIDITOK_TRAINING_ROLES,
    MidiTokRealTokenizer,
    MidiTokRoleSegment,
    MidiTokSource,
    MidiTokUnavailableError,
    export_miditok_role_dataset,
    load_miditok_segments,
)
from training.tokenizers.miditok_real import (
    MidiTokDatasetSummary as MidiTokRealDatasetSummary,
)
from training.tokenizers.symbolic import MidiTokBridgeTokenizer, build_miditok_bridge_config

__version__ = "0.1.0"

__all__ = [
    "BASELINE_ROLE_MODEL_TYPES",
    "CUSTOM_ROLE_NGRAM_MODEL_TYPE",
    "CUSTOM_ROLE_NGRAM_VERSION",
    "CUSTOM_ROLE_TRAINING_ROLES",
    "MIDITOK_TRAINING_ROLES",
    "TOKENIZATION_ROLES",
    "MidiTokBridgeTokenizer",
    "MidiTokRealDatasetSummary",
    "MidiTokRealTokenizer",
    "MidiTokRoleSegment",
    "MidiTokSource",
    "MidiTokUnavailableError",
    "CustomRoleNgramTrainingSummary",
    "RoleNgramCheckpoint",
    "RoleNgramModel",
    "RoleTrainingSegment",
    "StatisticalBaselineSummary",
    "StatisticalRoleModel",
    "StatisticalRoleModelArtifact",
    "TokenizedDatasetSummary",
    "TokenizedSegment",
    "TokenizedSegmentMetadata",
    "__version__",
    "build_miditok_bridge_config",
    "checkpoint_dir_for_role",
    "export_miditok_role_dataset",
    "export_tokenized_dataset",
    "load_miditok_segments",
    "load_role_training_segments",
    "load_tokenized_segments",
    "miditok_sources_from_dataset_manifest",
    "train_baseline_statistical_models",
    "train_custom_role_ngram_checkpoints",
]
