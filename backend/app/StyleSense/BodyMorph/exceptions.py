class BodyMorphError(Exception):
    """
    Base class for BodyMorph engine errors.

    All BodyMorph-specific exceptions should inherit from this type so callers
    (API layer, tests, workers) can catch a single umbrella exception when needed.
    """


class MultiSubjectError(BodyMorphError):
    """
    Raised when more than one person is detected.

    Typical causes:
      - Group photo
      - Reflections / posters / mannequins triggering detection
      - Duplicate/overlapping detections in the same frame
    """


class LowQualityError(BodyMorphError):
    """
    Raised when the image is too low quality to process.

    Typical causes:
      - Excessive blur / motion blur
      - Very low resolution / heavy compression artifacts
      - Poor lighting that destroys edge detail
    """


class HeavyOcclusionError(BodyMorphError):
    """
    Raised when occlusion prevents reliable estimation.

    Typical causes:
      - Large clothing layers obscuring silhouette
      - Objects blocking torso/hips (bags, furniture, people)
      - Cropped frames missing key regions
    """


class ExtremePerspectiveError(BodyMorphError):
    """
    Raised when perspective or camera pose is too extreme.

    Typical causes:
      - Wide-angle distortion (high FOV)
      - Very close camera distance
      - High tilt angles creating strong foreshortening
    """


class UnknownSilhouetteError(BodyMorphError):
    """
    Raised when the silhouette confidence is too low.

    Typical causes:
      - Ambiguous proportions near decision boundaries
      - Noisy landmarks/segmentation results
      - Conflicting signals due to quality/perspective issues
    """
