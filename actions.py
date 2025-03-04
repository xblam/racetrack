# file to store all of the global variables
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
CELL_SIZE = 40

ACTION_SPACE = ((-1, -1), (-1, 0), (-1, 1),
                (0, -1),  (0, 0),  (0, 1),
                (1, -1),  (1, 0),  (1, 1))

ACTION_TO_INDEX = {action: i for i, action in enumerate(ACTION_SPACE)}
INDEX_TO_ACTION = {i: action for action, i in ACTION_TO_INDEX.items()}

def action_to_index(action):
    return ACTION_TO_INDEX[action]

def index_to_action(index):
    return INDEX_TO_ACTION[index]