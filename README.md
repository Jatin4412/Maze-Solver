# 🧠 Q-Learning on 20×20 Emoji Mazes  
*A simple tabular Reinforcement Learning mini-project*

This project implements a **tabular Q-learning agent** that learns to navigate **20×20 grid mazes** represented using emojis. The setup is lightweight, easy to follow, and suitable as a student project for learning RL basics.

---

## 📌 Project Features

### ✅ Emoji-based maze design  
Mazes are written using emojis for readability:

- 🟦 = wall  
- ⬜ = free space  
- 🟩 = start (agent spawn)  
- 🟥 = goal (target)

These are automatically converted into a **numeric grid**:

| Emoji | Meaning | Code |
|-------|---------|------|
| ⬜ | Free cell | 0 |
| 🟦 | Wall | 1 |
| 🟩 | Start | 0 (with start coordinate) |
| 🟥 | Goal | 0 (with goal coordinate) |

---

### ✅ Four predefined 20×20 mazes
The project includes multiple maze types:

1. **open_with_bars** — open layout with horizontal bars  
2. **spiral** — alternating spiral walls  
3. **zigzag** — vertical zigzag corridors  
4. **random_dense** — 35% random walls with a carved workable path  

Each maze is **guaranteed solvable** via a randomized DFS path-carving function.

---

### ✅ Simple RL Environment (MazeEnv20)

The environment behaves like a small gym environment:

- **State** = flattened (row, col) → integer 0–399  
- **Actions (4):**
  - 0 = up  
  - 1 = down  
  - 2 = left  
  - 3 = right  
- **Episode ends when**:
  - agent reaches the goal  
  - or reaches max steps (default 1000)

### 🚦 Reward Structure

| Event | Reward |
|--------|--------|
| Step taken | −0.1 |
| Hit wall / invalid move | −0.5 |
| Reach goal | +10 |

This encourages shorter paths and penalizes unnecessary or invalid moves.

---

## 🤖 Q-Learning Agent

The agent uses a classic **tabular Q-learning** approach:

- Q-table shape: **400 × 4**
- Epsilon-greedy exploration  
- Hyperparameters:
  - learning rate (lr)
  - discount factor (gamma)
  - epsilon + decay  
- Q-update rule:
  ```
  Q[s,a] ← Q[s,a] + lr * (r + γ max(Q[s'],:) − Q[s,a])
  ```

---

## 🎯 Training Procedure

- Trains on **2500 episodes**
- Tracks:
  - total reward
  - steps per episode  
- Epsilon decays gradually  
- Shows a matplotlib reward plot  

---

## 📊 Evaluation

The trained agent is evaluated on **all four mazes**:

For each maze:
- success rate over 100 greedy rollouts  
- average steps taken  

Greedy paths are rendered visually:
- green = start  
- red = goal  
- black = walls  
- blue dots = path  

---

## ▶️ How to Run

Install dependencies:
```
pip install numpy matplotlib
```

Run:
```
python maze_qlearning.py
```

The script will automatically:
- print emoji mazes  
- train the Q-learning agent  
- plot rewards  
- evaluate on all mazes  
- render greedy paths  

---

## 📁 File Structure

```
maze_qlearning.py     # main project file
README.md              # this file
```

---

## 💡 Future Improvements

- Replace Q-table with Deep Q-Network (DQN)
- Add dynamic obstacles
- Larger mazes (e.g., 40×40)
- Multi-maze or curriculum training  
- Replay buffers or prioritized sampling  

---

Enjoy exploring Q-learning with emoji mazes 🚀🧩!
