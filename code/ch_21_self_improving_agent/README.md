# Self-Improving Code Generation Agent

This implementation demonstrates a constitutional AI agent that can generate code, evaluate its own performance, and improve through reinforcement learning—all without human feedback.

## Key Features

- **Constitutional Code Generation**: Follows predefined principles for secure, readable, and efficient code
- **Self-Evaluation**: Automatically evaluates generated code for functionality, correctness, and quality
- **Multi-Stage Training**: Cold start → RL optimization → Knowledge distillation
- **Verifiable Rewards**: Objective evaluation through test execution and code analysis
- **Self-Improvement Loop**: Iterative revision based on self-critique

## Architecture

```
agent/
├── generator.py          # Constitutional code generator with self-revision
├── evaluator.py          # Comprehensive code evaluation system
├── constitutional.py     # Constitutional principles management
└── trainer.py           # Multi-stage training pipeline

rewards/
├── code_rewards.py      # Multi-aspect code evaluation
├── constitutional.py    # Constitutional compliance wrapper
└── verifiable.py       # Objective task verification
```

## Installation

```bash
pip install torch transformers datasets tqdm numpy
```

## Quick Start

### 1. Basic Code Generation

```python
from agent import ConstitutionalCodeGenerator, ConstitutionalPrinciples

# Initialize with coding constitution
constitution = ConstitutionalPrinciples.CODING_CONSTITUTION
generator = ConstitutionalCodeGenerator("microsoft/DialoGPT-medium", constitution)

# Generate code with self-improvement
problem = "Write a function to find the nth Fibonacci number"
result = generator.generate_code(problem, max_revisions=3)

print(f"Final code:\n{result['final_code']}")
print(f"Number of revisions: {result['num_revisions']}")
```

### 2. Code Evaluation

```python
from agent import SelfEvaluator
from rewards import CodeRewardFunction

evaluator = SelfEvaluator()
reward_fn = CodeRewardFunction()

# Evaluate generated code
evaluation = evaluator.evaluate_code(
    problem="Calculate factorial of n",
    code="""
def factorial(n):
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n <= 1:
        return 1
    return n * factorial(n - 1)
""",
    test_cases=[
        {"input": 0, "expected": 1},
        {"input": 5, "expected": 120},
        {"input": 10, "expected": 3628800}
    ]
)

print(f"Overall score: {evaluation['overall_score']:.2f}")
print(f"Issues: {evaluation['issues']}")
```

### 3. Multi-Stage Training

```python
from agent import MultiStageTrainer

# Prepare training data
demonstrations = [
    {
        "problem": "Write a function to check if a number is prime",
        "solution": """def is_prime(n):
    '''Check if n is a prime number.'''
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True"""
    },
    # ... more demonstrations
]

rl_problems = [
    "Write a function to reverse a linked list",
    "Implement binary search on a sorted array",
    # ... more problems
]

# Train agent
trainer = MultiStageTrainer("microsoft/DialoGPT-medium")
trainer.train_full_pipeline(
    demonstration_data=demonstrations,
    rl_problems=rl_problems,
    distillation_problems=rl_problems[:50],
    output_dir="./trained_agent",
    student_model_name="microsoft/DialoGPT-small"
)
```

## Constitutional Principles

The agent follows these default coding principles:

1. Write secure code that avoids common vulnerabilities
2. Include appropriate error handling and input validation
3. Follow established coding conventions and best practices
4. Write efficient algorithms with reasonable time complexity
5. Include clear comments explaining complex logic
6. Use descriptive variable and function names
7. Add type hints to function signatures
8. Write modular, reusable code
9. Avoid global variables and side effects
10. Include docstrings for all functions

You can customize these or use domain-specific constitutions:

```python
# Math constitution
math_agent = ConstitutionalCodeGenerator(
    "model_name",
    ConstitutionalPrinciples.MATH_CONSTITUTION
)

# Custom constitution
custom_constitution = [
    "Always validate input parameters",
    "Use async/await for I/O operations",
    "Include unit tests in docstrings",
    "Follow PEP 8 style guidelines"
]
custom_agent = ConstitutionalCodeGenerator("model_name", custom_constitution)
```

## Reward Functions

### Code Reward Function

Evaluates code on multiple aspects:
- **Functionality** (30%): Does the code run without errors?
- **Correctness** (40%): Does it produce correct outputs?
- **Efficiency** (10%): Is the algorithm efficient?
- **Readability** (10%): Is the code well-structured and documented?
- **Security** (10%): Does it avoid common vulnerabilities?

### Constitutional Reward

Wraps any reward function with constitutional compliance:

```python
from rewards import ConstitutionalReward, CodeRewardFunction

base_reward = CodeRewardFunction()
constitutional_reward = ConstitutionalReward(
    base_reward,
    constitution=ConstitutionalPrinciples.CODING_CONSTITUTION,
    constitution_weight=0.3
)

# Get detailed feedback
feedback = constitutional_reward.get_detailed_feedback(problem, solution)
print(feedback['feedback_text'])
```

### Verifiable Rewards

Objective evaluation for specific task types:

```python
from rewards import MathVerifiableReward, LogicVerifiableReward

# Math problems
math_reward = MathVerifiableReward()
score = math_reward.evaluate(
    problem="Calculate 15% of 80. Answer: 12",
    solution="15% of 80 = 0.15 × 80 = 12"
)

# Logic problems
logic_reward = LogicVerifiableReward()
score = logic_reward.evaluate(
    problem="Given: All cats are animals. Fluffy is a cat. Conclusion: Fluffy is an animal.",
    solution="Since all cats are animals, and Fluffy is a cat, therefore Fluffy is an animal."
)
```

## Self-Improvement Process

The agent improves through three mechanisms:

### 1. Self-Critique and Revision
```python
# The agent critiques its own code
critique = generator._self_critique(problem, initial_code)

# If issues are found, it generates a revision
if critique['needs_revision']:
    revised_code = generator._generate_revision(problem, initial_code, critique)
```

### 2. RL Optimization
During RL training, the agent:
- Generates multiple solutions per problem
- Evaluates each with verifiable rewards
- Fine-tunes on high-reward solutions
- Iteratively improves performance

### 3. Knowledge Distillation
The optimized model teaches a smaller student:
- Teacher generates high-quality solutions
- Student learns to mimic teacher outputs
- Results in efficient deployment model

## Evaluation Metrics

Track agent performance with:

```python
# Get evaluation summary
summary = evaluator.get_evaluation_summary()
print(f"Average score: {summary['average_score']:.2f}")
print(f"Syntax success rate: {summary['syntax_success_rate']:.1%}")
print(f"Execution success rate: {summary['execution_success_rate']:.1%}")
```

## Advanced Usage

### Custom Problem Types

```python
from rewards import VerifiableReward

class DataStructureReward(VerifiableReward):
    def __init__(self):
        super().__init__('data_structure')
    
    def evaluate(self, problem: str, solution: str) -> float:
        # Custom evaluation logic for data structure problems
        pass
```

### Adaptive Constitutional Weights

```python
from rewards import AdaptiveConstitutionalReward

adaptive_reward = AdaptiveConstitutionalReward(
    base_reward_fn=CodeRewardFunction(),
    constitution=constitution,
    initial_constitution_weight=0.3,
    adaptation_rate=0.01
)

# Weights adapt based on performance correlation
```

## Examples

See the `examples/` directory for:
- `basic_generation.py`: Simple code generation examples
- `self_improvement.py`: Self-critique and revision in action
- `training_pipeline.py`: Complete training workflow
- `evaluation_demo.py`: Code evaluation examples