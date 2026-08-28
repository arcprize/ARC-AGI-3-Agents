def choose_action(actions, state, model):
    best_action = None
    best_score = -999

    for a in actions:
        future = model.predict(state, a)

        score = 0
        if not future.get("danger"):
            score += 10

        if future.get("goal_visible"):
            score += 20

        if score > best_score:
            best_score = score
            best_action = a

    return best_action
