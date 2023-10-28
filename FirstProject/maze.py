import numpy as np
import random
import pygame

#Define the maze we'll be using.
maze = np.array([
    [0, 0, 0, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0],
    [0, 0, 0, 0, 2]
])

#this maze requires exploration prob to be 0.5 
# maze = np.array([
#     [0, 0, 0, 0, 2],
#     [0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0],
#     [0, 0, 0, 0, 0]
# ])

#defines the parameters of the maze
num_states = maze.size
num_actions = 4
learning_rate = 0.1
discount_factor = 0.95
exploration_prob = 0.5

#initialize the q_table
q_table = np.zeros((num_states, num_actions))

#defines the rewards for each state
rewards = {
    'free': 0.01,
    'goal': 10.0,
    'occupied': -0.5,
    'visited': -0.5,
    'lost': -1.0

}

#the maze is a 5x5 grid, so the goal state is the first 2 in the last row
#goal_state = np.argwhere(maze == 2)[0][0]

goal_state = 24

#defines the valid actions for each state
def get_valid_actions(state):
    # Get the x and y coordinates of the state
    x, y = divmod(state, maze.shape[1]) 
    valid_actions = [] 

    # Check if the state is not at the top edge of the maze and the state above is not occupied
    if x > 0 and maze[x - 1, y] != 1:
        valid_actions.append(0) 
    # Check if the state is not at the bottom edge of the maze and the state below is not occupied
    if x < maze.shape[0] - 1 and maze[x + 1, y] != 1:
        valid_actions.append(1)
    # Check if the state is not at the left edge of the maze and the state to the left is not occupied
    if y > 0 and maze[x, y - 1] != 1:
        valid_actions.append(2)
    # Check if the state is not at the right edge of the maze and the state to the right is not occupied
    if y < maze.shape[1] - 1 and maze[x, y + 1] != 1:
        valid_actions.append(3)

    return valid_actions

num_episodes = 1000 #1000

pygame.init()

# Define colors we'll use for tiles, etc.
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)  # Red color for agents
GREEN = (0, 255, 0)  # Green color for the goal
BLUE = (0, 0, 255)  # Blue color for the starting location

#pygame window settings
screen_size = (800, 800)
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption("Maze Navigation")

#Render the maze and agent's path
def render_maze(agent_path, start_state):
    screen.fill(WHITE)
    cell_size = screen_size[0] // maze.shape[0]

    # Render the maze
    for x in range(maze.shape[0]):
        for y in range(maze.shape[1]):
            if maze[x, y] == 1:
                pygame.draw.rect(screen, BLACK, (y * cell_size, x * cell_size, cell_size, cell_size))
            if maze[x, y] == 2:
                pygame.draw.rect(screen, GREEN, (y * cell_size, x * cell_size, cell_size, cell_size))
    
    # Render the starting location in blue
    start_x, start_y = divmod(start_state, maze.shape[1])
    start_rect = pygame.Rect(start_y * cell_size, start_x * cell_size, cell_size, cell_size)
    pygame.draw.rect(screen, BLUE, start_rect)

    # Render the agent's path in red 
    for state in agent_path:
        x, y = divmod(state, maze.shape[1])
        agent_rect = pygame.Rect(y * cell_size + cell_size // 4, x * cell_size + cell_size // 4, cell_size // 2, cell_size // 2)
        pygame.draw.rect(screen, RED, agent_rect)

    pygame.display.flip() #updates pygame display

#Store the best path and its cumulative reward
best_path = []
#best_cumulative_reward = float('-inf') 
best_cumulative_reward = -np.inf

#stop threshold
#stop_threshold = (0.01 * 8) + 10.0
#stop_threshold = 6

# Training loop
for episode in range(num_episodes):
    # Initialize state and path
    start_state = 0  # Define the starting state at the top left corner (0, 0)
    state = start_state
    agent_path = [state]
    done = False

    while not done:
        # Choose a random action or the best action based on the Q-table
        if random.uniform(0, 1) < exploration_prob:  #if the random number is less than the exploration probability, choose a random action
            action = random.choice(get_valid_actions(state)) #either up, down, left, or right
        else:
            action = np.argmax(q_table[state]) 

        x, y = divmod(state, maze.shape[1])  # Get the x and y coordinates of the state

        # Move the agent based on the action chosen (0=up, 1=down, 2=left, 3=right)
        if action == 0:
            x -= 1
        elif action == 1:
            x += 1
        elif action == 2:
            y -= 1
        else:
            y += 1

        # Check if the new position is outside the boundaries
        x = max(0, min(x, maze.shape[0] - 1))
        y = max(0, min(y, maze.shape[1] - 1))

        new_state = x * maze.shape[1] + y # Get the new state based on the new x and y positions

        if maze[x, y] == 1: # Check if the new position is occupied
            reward = rewards['occupied']
        elif maze[x, y] == 2: # Check if the goal is reached
            reward = rewards['goal']
            
        elif new_state in agent_path: # Check if the new position has already been visited
            reward = rewards['visited']
        else: # Otherwise, it's a free cell
            reward = rewards['free']

        

        # Update the Q-table
        q_table[state, action] = (1 - learning_rate) * q_table[state, action] + learning_rate * (reward + discount_factor * np.max(q_table[new_state]))

        #update state and path 
        state = new_state
        agent_path.append(state)

        # Check if the new state is the goal state
        if state == goal_state:
            done = True

        #if we are in a loop, we are lost
        if reward == rewards['visited']:
            reward = rewards['lost']
            done = True

        # Update the Pygame window to visualize the agent's movements
        render_maze(agent_path, start_state)
        pygame.time.delay(100)  # Delay to slow down visualization


    cumulative_reward = sum([rewards['goal'] if s == goal_state else rewards['visited'] for s in agent_path])
    print(f"Episode {episode + 1}, Cumulative Reward: {cumulative_reward}")


    # Check if the current path is the best path
    if cumulative_reward > best_cumulative_reward:
        best_path = agent_path
        best_cumulative_reward = cumulative_reward



    # Check if you want to exit the loop when a better path is found
    # if best_cumulative_reward >= stop_threshold:
    #     break


#save the best path that it took 
def visualize_best_path(best_path, start_state):
    pygame.init()
    screen_size = (800, 800)
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("Maze Navigation")
    cell_size = screen_size[0] // maze.shape[0]
    screen.fill(WHITE)

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Render the maze
        for x in range(maze.shape[0]):
            for y in range(maze.shape[1]):
                if maze[x, y] == 1:
                    pygame.draw.rect(screen, BLACK, (y * cell_size, x * cell_size, cell_size, cell_size))
                if maze[x, y] == 2:
                    pygame.draw.rect(screen, GREEN, (y * cell_size, x * cell_size, cell_size, cell_size))

        # Render the starting location in blue
        start_x, start_y = divmod(start_state, maze.shape[1])
        start_rect = pygame.Rect(start_y * cell_size, start_x * cell_size, cell_size, cell_size)
        pygame.draw.rect(screen, BLUE, start_rect)

        # Render the best path in red
        for state in best_path:
            x, y = divmod(state, maze.shape[1])
            agent_rect = pygame.Rect(y * cell_size + cell_size // 4, x * cell_size + cell_size // 4, cell_size // 2, cell_size // 2)
            pygame.draw.rect(screen, RED, agent_rect)

        pygame.display.flip()

    pygame.quit()    
    

visualize_best_path(best_path, start_state)


