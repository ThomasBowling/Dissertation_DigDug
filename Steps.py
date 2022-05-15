from Actions import Actions

# Waits certain amount of time then commits and action in order to quickly start a game and speeds up the starting animation
def start_game(frame_ratio):
    return [
        {"wait": int(180/frame_ratio), "actions": [Actions.COIN]},
        {"wait": int(60/frame_ratio), "actions": [Actions.START]},
        {"wait": int(534/frame_ratio), "actions": [Actions.START]}]

