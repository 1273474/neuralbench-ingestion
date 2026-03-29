"""Pydantic response models for the TRIBE v2 inference API."""

from pydantic import BaseModel, Field


class AnalysisResult(BaseModel):
    """
    Cognitive load metrics derived from TRIBE v2 fMRI voxel
    prediction, mapped to three anatomical brain regions.
    """
    visual_intensity: float = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Attention score derived from Superior Colliculus + Visual Cortex "
            "(V1-V5) voxel activation. 0 = no visual salience, 100 = maximum "
            "saccadic and motion-processing load."
        ),
    )
    audio_complexity: float = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Desire/Emotional Arousal score derived from Ventral Striatum "
            "(Nucleus Accumbens) + Amygdala voxel activation. 0 = no hedonic "
            "response, 100 = maximum reward-circuit engagement."
        ),
    )
    text_density: float = Field(
        ...,
        ge=0,
        le=100,
        description=(
            "Memory Encoding score derived from Hippocampus + Parahippocampal "
            "Gyrus voxel activation. 0 = no memory load, 100 = maximum "
            "episodic encoding demand."
        ),
    )
