with open('utils.py', 'r') as f:
    content = f.read()

# Remove the dead commented imports
content = content.replace("# # import torchaudio  # For saving PyTorch tensors and potentially speed adjustment.\n", "")
content = content.replace("# import torch\n", "")
content = content.replace("    import torchaudio\n    import torch\n", "")

# Fix the docstrings order and the raise RuntimeError
func1_old = """def save_audio_tensor_to_file(
    audio_tensor: 'torch.Tensor',
    sample_rate: int,
    file_path_str: str,
    output_format: str = "wav",
) -> bool:
    try:
        import torch
        import torchaudio
    except ImportError as e:
        logger.error(f"Failed to import torch/torchaudio: {e}")
        return False

    \"\"\"
    Saves a PyTorch audio tensor to a file using torchaudio.

    Args:
        audio_tensor: PyTorch tensor containing audio data.
        sample_rate: Sample rate of the audio data.
        file_path_str: String path to save the audio file.
        output_format: Desired output format (passed to torchaudio.save).

    Returns:
        True if saving was successful, False otherwise.
    \"\"\""""

func1_new = """def save_audio_tensor_to_file(
    audio_tensor: 'torch.Tensor',
    sample_rate: int,
    file_path_str: str,
    output_format: str = "wav",
) -> bool:
    \"\"\"
    Saves a PyTorch audio tensor to a file using torchaudio.

    Args:
        audio_tensor: PyTorch tensor containing audio data.
        sample_rate: Sample rate of the audio data.
        file_path_str: String path to save the audio file.
        output_format: Desired output format (passed to torchaudio.save).

    Returns:
        True if saving was successful, False otherwise.
    \"\"\"
    try:
        import torch
        import torchaudio
    except ImportError as e:
        raise RuntimeError(f"torchaudio is required to save audio tensors. {e}")
"""

content = content.replace(func1_old, func1_new)

func2_old = """def apply_speed_factor(
    audio_tensor: 'torch.Tensor', sample_rate: int, speed_factor: float
) -> Tuple['torch.Tensor', int]:
    try:
        import torch
        import torchaudio
    except ImportError as e:
        logger.error(f"Failed to import torch/torchaudio: {e}")
        return audio_tensor, sample_rate

    \"\"\"
    Applies a speed factor to an audio tensor.
    Uses librosa.effects.time_stretch if available for pitch preservation.
    Falls back to simple resampling via torchaudio.transforms.Resample if librosa is not available,
    which will alter pitch.

    Args:
        audio_tensor: Input audio waveform (PyTorch tensor, expected mono).
        sample_rate: Sample rate of the input audio.
        speed_factor: Desired speed factor (e.g., 1.0 is normal, 1.5 is faster, 0.5 is slower).

    Returns:
        A tuple of the speed-adjusted audio tensor and its sample rate (which remains unchanged).
        Returns the original tensor and sample rate if speed_factor is 1.0 or if adjustment fails.
    \"\"\""""

func2_new = """def apply_speed_factor(
    audio_tensor: 'torch.Tensor', sample_rate: int, speed_factor: float
) -> Tuple['torch.Tensor', int]:
    \"\"\"
    Applies a speed factor to an audio tensor.
    Uses librosa.effects.time_stretch if available for pitch preservation.
    Falls back to simple resampling via torchaudio.transforms.Resample if librosa is not available,
    which will alter pitch.

    Args:
        audio_tensor: Input audio waveform (PyTorch tensor, expected mono).
        sample_rate: Sample rate of the input audio.
        speed_factor: Desired speed factor (e.g., 1.0 is normal, 1.5 is faster, 0.5 is slower).

    Returns:
        A tuple of the speed-adjusted audio tensor and its sample rate (which remains unchanged).
        Returns the original tensor and sample rate if speed_factor is 1.0 or if adjustment fails.
    \"\"\"
    try:
        import torch
        import torchaudio
    except ImportError as e:
        raise RuntimeError(f"torchaudio is required to apply speed factor. {e}")
"""

content = content.replace(func2_old, func2_new)

with open('utils.py', 'w') as f:
    f.write(content)
