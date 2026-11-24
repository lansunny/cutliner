"""FFmpeg geometry for model inputs and reproducible legacy controls."""

FIT_PROFILE = "fit_dar224_v1"


def square_geometry(view: str, *, bicubic=False) -> tuple[str, str]:
    """Return a scale filter and a post-color filter, without altering timing."""
    flags = ":flags=bicubic" if bicubic else ""
    if view == "center":
        # Retained only for cached experiments and explicit legacy comparisons.
        return f"scale=224:224{flags}:force_original_aspect_ratio=increase,crop=224:224", ""
    if view != "fit":
        raise ValueError(f"Unknown frame geometry: {view}")
    # FFmpeg dar includes sample aspect ratio. Round down to an even size
    # for 4:2:0 inputs, then set square pixels before padding to the model size.
    scale = ("scale=w='if(gte(dar,1),224,max(2,2*trunc(112*dar)))'"
             ":h='if(gte(dar,1),max(2,2*trunc(112/dar)),224)'"
             + flags + ",setsar=1")
    return scale, ",pad=224:224:(ow-iw)/2:(oh-ih)/2:color=0x808080"
