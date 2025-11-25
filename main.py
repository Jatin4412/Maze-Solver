"""
Q-learning on 20x20 mazes — simple tabular implementation.

Emoji maze legend (used in prompts / comments):
🟦 = wall
⬜ = free
🟩 = start (S)
🟥 = goal (G)

This file contains:
- several predefined 20x20 mazes (emoji visual in comments)
- converter from emoji -> numeric grid (0 free, 1 wall)
- MazeEnv20: gym-like env with 4 actions
- Q-learning agent (tabular)
- training loop (2000-3000 episodes)
- evaluation and greedy-path visualization (matplotlib)

Run as a script. It will train and show simple plots and greedy rollouts.

Designed to be a compact, student-friendly RL mini-project.
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import random

# ----------------------------
# Helpers: emoji <-> numeric
# ----------------------------
EMOJI_WALL = '🟦'
EMOJI_FREE = '⬜'
EMOJI_START = '🟩'
EMOJI_GOAL = '🟥'


def emoji_to_grid(emoji_rows):
    """
    Convert list of 20 strings (each string has 20 emoji characters) to numeric grid.
    Returns grid (20,20) with 0=free,1=wall and start,goal coordinates.
    """
    H = len(emoji_rows)
    W = len(emoji_rows[0])
    grid = np.zeros((H, W), dtype=np.int8)
    start = None
    goal = None
    for r, row in enumerate(emoji_rows):
        # row is a string where each emoji is a single unicode character
        cells = list(row)
        for c, ch in enumerate(cells):
            if ch == EMOJI_WALL:
                grid[r, c] = 1
            elif ch == EMOJI_FREE:
                grid[r, c] = 0
            elif ch == EMOJI_START:
                grid[r, c] = 0
                start = (r, c)
            elif ch == EMOJI_GOAL:
                grid[r, c] = 0
                goal = (r, c)
            else:
                # unknown -> treat as free
                grid[r, c] = 0
    if start is None or goal is None:
        raise ValueError('Start or goal not found in emoji maze.')
    return grid, start, goal


# ----------------------------
# Maze creation utilities
# ----------------------------

def carve_path_random(grid, start, goal, seed=None):
    """
    Carve a guaranteed path between start and goal using randomized DFS, marking path cells as free.
    This is used if we create a mostly-walled maze but want to ensure solvability.
    """
    rng = random.Random(seed)
    H, W = grid.shape
    visited = set()
    path = []

    def neighbors(cell):
        r, c = cell
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < H and 0 <= nc < W:
                yield (nr, nc)

    stack = [start]
    parent = {start: None}
    found = False
    while stack:
        node = stack.pop()
        if node == goal:
            found = True
            break
        if node in visited:
            continue
        visited.add(node)
        nbrs = list(neighbors(node))
        rng.shuffle(nbrs)
        for nb in nbrs:
            if nb not in visited:
                parent[nb] = node
                stack.append(nb)
    if not found:
        return grid
    # reconstruct path
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    for (r,c) in path:
        grid[r,c] = 0
    return grid


def make_maze_empty():
    """Simple open maze with border walls."""
    H = W = 20
    grid = np.zeros((H,W), dtype=np.int8)
    grid[0,:] = 1
    grid[-1,:] = 1
    grid[:,0] = 1
    grid[:,-1] = 1
    start = (1,1)
    goal = (18,18)
    # add some obstacles
    for r in range(3,17):
        if r % 2 == 0:
            grid[r, 5:15] = 1
    # carve a couple of gaps to ensure path
    grid[4,9] = 0
    grid[6,7] = 0
    grid[10,12] = 0
    return grid, start, goal


def make_maze_spiral():
    H = W = 20
    grid = np.ones((H,W), dtype=np.int8)
    r0 = c0 = 1
    r1 = c1 = 18
    val = 0
    while r0 <= r1 and c0 <= c1:
        # top
        grid[r0, c0:c1+1] = val
        # right
        grid[r0:r1+1, c1] = val
        # bottom
        grid[r1, c0:c1+1] = val
        # left
        grid[r0:r1+1, c0] = val
        r0 += 2; c0 += 2; r1 -= 2; c1 -= 2
        val = 1 - val
    start = (1,1)
    goal = (18,18)
    # make sure start/goal free
    grid[start] = 0
    grid[goal] = 0
    # Carve path guarantee
    grid = carve_path_random(grid, start, goal, seed=42)
    return grid, start, goal


def make_maze_zigzag():
    H = W = 20
    grid = np.zeros((H,W), dtype=np.int8)
    # border walls
    grid[0,:] = 1
    grid[-1,:] = 1
    grid[:,0] = 1
    grid[:,-1] = 1
    # vertical walls with single gaps creating zigzag
    for c in range(2,18,2):
        grid[2:18, c] = 1
        # gap position
        gap = 2 + (c % 6)
        grid[gap, c] = 0
    start = (1,1)
    goal = (18,18)
    grid = carve_path_random(grid, start, goal, seed=7)
    return grid, start, goal


def make_maze_random_dense(seed=0):
    """Create a mostly random maze but ensure path by carving."""
    rng = np.random.RandomState(seed)
    H = W = 20
    grid = (rng.rand(H,W) < 0.35).astype(np.int8)  # 35% walls
    # force borders
    grid[0,:] = 1; grid[-1,:] = 1; grid[:,0] = 1; grid[:,-1] = 1
    start = (1,1)
    goal = (18,18)
    # carve guaranteed path
    grid = carve_path_random(grid, start, goal, seed=seed)
    return grid, start, goal


# ----------------------------
# Convert numeric grid -> emoji lines (for printing)
# ----------------------------

def grid_to_emoji_lines(grid, start, goal):
    H,W = grid.shape
    lines = []
    for r in range(H):
        row = []
        for c in range(W):
            if (r,c) == start:
                row.append(EMOJI_START)
            elif (r,c) == goal:
                row.append(EMOJI_GOAL)
            elif grid[r,c] == 1:
                row.append(EMOJI_WALL)
            else:
                row.append(EMOJI_FREE)
        lines.append(''.join(row))
    return lines


# ----------------------------
# Environment
# ----------------------------

class MazeEnv20:
    def __init__(self, grid, start, goal, max_steps=1000):
        self.grid = grid.copy()
        self.start = start
        self.goal = goal
        self.h, self.w = grid.shape
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        self.agent_pos = self.start
        self.steps = 0
        return self._state()

    def _state(self):
        r, c = self.agent_pos
        return r * self.w + c

    def step(self, action):
        # action: 0 up, 1 down, 2 left, 3 right
        drc = [(-1,0),(1,0),(0,-1),(0,1)]
        dr, dc = drc[action]
        nr = self.agent_pos[0] + dr
        nc = self.agent_pos[1] + dc
        hit_wall = False
        # boundary or wall => stay in place
        if not (0 <= nr < self.h and 0 <= nc < self.w) or self.grid[nr, nc] == 1:
            nr, nc = self.agent_pos
            hit_wall = True
        self.agent_pos = (nr, nc)
        self.steps += 1
        done = (self.agent_pos == self.goal)
        reward = -0.1  # step penalty
        if hit_wall:
            reward += -0.5
        if done:
            reward += 10.0
        if self.steps >= self.max_steps:
            done = True
        return self._state(), reward, done, {'pos': self.agent_pos}

    def render(self, path=None, figsize=(6,6)):
        # path: list of (r,c)
        grid = self.grid
        H,W = grid.shape
        display = np.zeros((H,W,3), dtype=np.float32)
        # default white/free
        display[:,:,:] = 1.0
        # walls black
        display[grid==1] = 0.0
        # start green
        sr, sc = self.start
        display[sr, sc] = np.array([0.6, 1.0, 0.6])
        # goal red
        gr, gc = self.goal
        display[gr, gc] = np.array([1.0, 0.6, 0.6])
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(display, interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        if path is not None and len(path) > 0:
            ys = [p[1] for p in path]
            xs = [p[0] for p in path]
            # plot path as blue dots
            ax.plot(ys, xs, marker='o', linewidth=2, markersize=4)
        ax.set_title('Maze (start green, goal red, walls black)')
        plt.show()


# ----------------------------
# Q-learning agent & training
# ----------------------------

class QLearner:
    def __init__(self, env, lr=0.5, gamma=0.99, epsilon=1.0, eps_min=0.05, eps_decay=0.9995):
        self.env = env
        self.n_states = env.h * env.w
        self.n_actions = 4
        self.Q = np.zeros((self.n_states, self.n_actions), dtype=np.float32)
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_decay = eps_decay

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, s, a, r, s2, done):
        target = r
        if not done:
            target += self.gamma * np.max(self.Q[s2])
        self.Q[s,a] += self.lr * (target - self.Q[s,a])

    def decay_epsilon(self):
        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)


# ----------------------------
# Training loop
# ----------------------------

def train_agent(env, episodes=2500, max_steps_per_ep=1000, **agent_kwargs):
    agent = QLearner(env, **agent_kwargs)
    rewards = []
    steps_list = []
    for ep in range(episodes):
        s = env.reset()
        done = False
        total_r = 0.0
        steps = 0
        while not done and steps < max_steps_per_ep:
            a = agent.choose_action(s)
            s2, r, done, info = env.step(a)
            agent.update(s, a, r, s2, done)
            s = s2
            total_r += r
            steps += 1
        agent.decay_epsilon()
        rewards.append(total_r)
        steps_list.append(steps)
        if (ep+1) % 250 == 0:
            print(f"Episode {ep+1}/{episodes}, reward={total_r:.2f}, steps={steps}, eps={agent.epsilon:.3f}")
    return agent, rewards, steps_list


# ----------------------------
# Greedy rollout and evaluation
# ----------------------------

def greedy_rollout(env, agent, max_steps=1000):
    s = env.reset()
    path = [env.agent_pos]
    done = False
    steps = 0
    while not done and steps < max_steps:
        a = int(np.argmax(agent.Q[s]))
        s, r, done, info = env.step(a)
        path.append(env.agent_pos)
        steps += 1
    return path, done, steps


def evaluate_agent(env, agent, episodes=50):
    success = 0
    steps_list = []
    for _ in range(episodes):
        _, done, steps = greedy_rollout(env, agent)
        if done:
            success += 1
        steps_list.append(steps)
    return success / episodes, np.mean(steps_list)


# ----------------------------
# Main: build mazes, train, evaluate, visualize
# ----------------------------

def main():
    # create 4 predefined mazes
    mazes = []
    g1, s1, t1 = make_maze_empty()
    mazes.append((g1, s1, t1, 'open_with_bars'))
    g2, s2, t2 = make_maze_spiral()
    mazes.append((g2, s2, t2, 'spiral'))
    g3, s3, t3 = make_maze_zigzag()
    mazes.append((g3, s3, t3, 'zigzag'))
    g4, s4, t4 = make_maze_random_dense(seed=3)
    mazes.append((g4, s4, t4, 'random_dense'))

    # print emoji versions for the user (20x20 each)
    print('\n--- Emoji mazes (20x20) ---')
    for i,(grid,start,goal,name) in enumerate(mazes):
        print(f'\nMaze {i+1}: {name}')
        lines = grid_to_emoji_lines(grid, start, goal)
        for line in lines:
            print(line)

    # Train on maze 1 (index 0)
    train_idx = 0
    env_train = MazeEnv20(mazes[train_idx][0], mazes[train_idx][1], mazes[train_idx][2], max_steps=1000)
    print('\nTraining Q-learning agent on maze 1 (open_with_bars)')
    agent, rewards, steps_list = train_agent(env_train, episodes=2500, lr=0.6, gamma=0.99, epsilon=1.0, eps_min=0.05, eps_decay=0.9996)

    # simple reward plot
    plt.figure(figsize=(8,3))
    plt.plot(rewards)
    plt.title('Episode total reward')
    plt.xlabel('Episode')
    plt.ylabel('Total reward')
    plt.tight_layout()
    plt.show()

    # Evaluate on all mazes
    print('\nEvaluation (greedy policy) on all mazes:')
    for i,(grid,start,goal,name) in enumerate(mazes):
        env = MazeEnv20(grid, start, goal, max_steps=1000)
        succ_rate, avg_steps = evaluate_agent(env, agent, episodes=100)
        print(f' Maze {i+1} ({name}): success_rate={succ_rate*100:.1f}%, avg_steps={avg_steps:.1f}')

    # show greedy path on training maze
    print('\nGreedy rollout path on training maze:')
    path, done, steps = greedy_rollout(env_train, agent)
    print(f' completed={done}, steps={steps}')
    env_train.render(path)

    # show greedy path on one other maze (index 2)
    print('\nGreedy rollout path on maze 3 (zigzag):')
    env_other = MazeEnv20(mazes[2][0], mazes[2][1], mazes[2][2])
    path2, done2, steps2 = greedy_rollout(env_other, agent)
    print(f' completed={done2}, steps={steps2}')
    env_other.render(path2)


if __name__ == '__main__':
    main()
