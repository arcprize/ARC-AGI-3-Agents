def compute_reward(info):
    """
    Converts game feedback into learning signal.
    """

    reward = 0

    if info.get("score"):
        reward += info["score"]

    if info.get("dead"):
        reward -= 10

    if info.get("progress"):
        reward += 2

    return reward
