# Practical Reinforcement Learning

**This book is a work in progress. Feel free to file an issue when you encounter any ambiguity in text or code.**

A comprehensive on practical reinforcement learning that bridges the gap between foundational theory and real-world applications. This book covers everything from classical RL methods to cutting-edge techniques in LLM training and production deployment.

## Book Structure

### Part 1: The Language of Learning
- **[Chapter 1: The Reinforcement Learning Paradigm](book/part_1_foundations/ch_01_the_rl_paradigm.md)**
- **[Chapter 2: The Formal Framework: Markov Decision Processes](book/part_1_foundations/ch_02_markov_decision_processes.md)**

### Part 2: Classical Toolkits for Small Worlds
- **[Chapter 3: Planning with a Perfect Model: Dynamic Programming](book/part_2_classical_toolkits/ch_03_dynamic_programming.md)**
- **[Chapter 4: Learning from Experience: MC and TD](book/part_2_classical_toolkits/ch_04_mc_and_td.md)**
- **[Chapter 5: The Cornerstone Algorithm: A Deep Dive into Q-Learning](book/part_2_classical_toolkits/ch_05_q_learning_deep_dive.md)**

### Part 3: Scaling Intelligence with Deep Learning
- **Chapter 6**: From Tables to Tensors: Deep Q-Networks (DQN)
- **Chapter 7**: The DQN Zoo: A Tour of Key Improvements
- **Chapter 8**: Learning the Strategy: Policy Gradient Methods
- **Chapter 9**: The Symbiosis: Actor-Critic Methods

### Part 4: The Practitioner's Crucible: From Theory to Reality
- **Chapter 10**: The Art and Science of Reward Design
- **Chapter 11**: Language as a Guide: Using LLMs in Reinforcement Learning
- **Chapter 12**: The Perennial Challenge: A Deep Dive into Exploration
- **Chapter 13**: Bridging the Reality Gap: The Sim2Real Problem
- **Chapter 14**: Learning from the Past: Offline Reinforcement Learning
- **Chapter 15**: When Things Go Wrong: Debugging and Diagnosing RL Systems

### Part 5: The Modern Frontier: State-of-the-Art Algorithms
- **Chapter 16**: Proximal Policy Optimization (PPO)
- **Chapter 17**: Soft Actor-Critic (SAC)
- **Chapter 18**: An Introduction to Multi-Agent RL (MARL)

### Part 6: Reinforcement Learning for Language Models and AI Agents
- **[Chapter 19: Reinforcement Learning from Human Feedback (RLHF)](book/part_6_rl_for_llms/ch_19_rlhf.md)**
- **[Chapter 20: Group Relative Policy Optimization (GRPO): The DeepSeek Revolution](book/part_6_rl_for_llms/ch_20_grpo.md)**
- **Chapter 21**: Reinforcement Fine-Tuning for AI Agents
- **Chapter 22**: Advanced Topics in LLM Training with RL

### Part 7: The Future of Practical RL
- **Chapter 23**: Emerging Paradigms and Research Frontiers
- **Chapter 24**: Production RL: From Research to Real-World Deployment

### Appendices
- **Appendix A**: Glossary of Terms and Notation
- **Appendix B**: A Primer on PyTorch for RL
- **Appendix C**: Modern RL Libraries and Tools
- **Appendix D**: Mathematical Prerequisites and Reference Formulas

## Code Implementation

Each chapter with practical implementations includes complete, production-quality code in the `code/` directory. Key implementations include:

- **Value Iteration** for gridworld environments
- **Q-Learning** from scratch
- **Deep Q-Networks (DQN)** for CartPole
- **REINFORCE** with baseline
- **Actor-Critic (A2C)** agent
- **Proximal Policy Optimization (PPO)** for continuous control
- **Soft Actor-Critic (SAC)** for robotics
- **RLHF Pipeline** with transformers
- **GRPO Implementation** for mathematical reasoning
- **Self-Improving Code Agent** with constitutional AI
- **Multi-Stage RL Pipeline** for reasoning models
- **Production RL System** with safety and monitoring

## Key Features

1. **Depth Over Breadth**: Each chapter provides comprehensive coverage of its topic, including mathematical foundations, implementation details, and practical considerations.

2. **Visual Learning**: Extensive use of Mermaid diagrams to illustrate complex concepts, architectures, and algorithms.

3. **Production-Ready Code**: All implementations follow software engineering best practices with proper documentation, testing considerations, and scalability in mind.

4. **Modern Coverage**: Includes cutting-edge topics like RLHF, GRPO, and production deployment that are often missing from traditional RL texts.

5. **Practical Focus**: Emphasizes real-world application with dedicated chapters on debugging, reward design, and the sim-to-real gap.

## Prerequisites

- **Mathematics**: Linear algebra, calculus, probability theory (covered in Appendix D)
- **Programming**: Python proficiency, basic PyTorch knowledge (primer in Appendix B)
- **Machine Learning**: Basic understanding of supervised learning helpful but not required

## Getting Started

1. **For Beginners**: Start with Part 1 for foundations, then progress through Parts 2-3
2. **For Practitioners**: Jump to Part 4 for practical considerations, or Part 5 for modern algorithms
3. **For LLM Researchers**: Focus on Part 6 for RLHF and related techniques
4. **For Engineers**: Part 7 covers production deployment and emerging paradigms

## Running the Code

Each code directory includes its own README with specific instructions. General setup:

```bash
# Create virtual environment
python -m venv rl_env
source rl_env/bin/activate  # On Windows: rl_env\Scripts\activate

# Install base dependencies
pip install torch gymnasium numpy matplotlib tqdm

# For specific chapters, install additional requirements
cd code/ch_XX_name/
pip install -r requirements.txt  # If available

# Run examples
python main.py  # or specific script name
```

## Contributing

This book represents a comprehensive resource for the RL community. While it's designed as a complete reference, suggestions for improvements, corrections, or additional topics are welcome.

## Citation

If you find this book helpful in your research or work, please cite:

```bibtex
@book{practical_rl_2025,
  title={Practical Reinforcement Learning},
  author={Obinna Okechukwu},
  year={2025},
  publisher={[Publisher]}
}
```

## License

This work is intended for educational and research purposes. Please refer to individual code licenses where applicable.

---

*"The best way to learn reinforcement learning is to implement it, debug it, and deploy it. This book guides you through all three."*