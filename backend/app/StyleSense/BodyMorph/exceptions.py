class BodyMorphError(Exception):
    """Base class for BodyMorph engine errors."""


class MultiSubjectError(BodyMorphError):
    """Raised when more than one person is detected."""


class LowQualityError(BodyMorphError):
    """Raised when the image is too low quality to process."""


class HeavyOcclusionError(BodyMorphError):
    """Raised when occlusion prevents reliable estimation."""


class ExtremePerspectiveError(BodyMorphError):
    """Raised when perspective or camera pose is too extreme."""


class UnknownSilhouetteError(BodyMorphError):
    """Raised when the silhouette confidence is too low."""
