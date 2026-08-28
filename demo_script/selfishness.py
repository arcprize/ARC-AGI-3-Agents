def encode(obs):
    """
    Convert raw environment observation into structured features.
    Replace this with real parsing from ARC engine output.
    """

    return {
        "raw": str(obs),
        "danger": "enemy" in str(obs),
        "goal_visible": "goal" in str(obs),
        "open_space": True
    }
