"""
Complete example of the reasoning model development pipeline.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline import ReasoningModelPipeline, ReasoningPipelineConfig
import json
import numpy as np


def create_math_reward_function():
    """Create a reward function for mathematical reasoning."""
    
    def extract_answer(solution: str) -> str:
        """Extract numerical answer from solution."""
        import re
        
        # Look for patterns like "= 42" or "answer: 42" or "is 42"
        patterns = [
            r'=\s*([-+]?\d*\.?\d+)',
            r'answer[:\s]+\s*([-+]?\d*\.?\d+)',
            r'is\s+([-+]?\d*\.?\d+)',
            r'equals\s+([-+]?\d*\.?\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, solution.lower())
            if match:
                return match.group(1)
        
        # Look for last number in solution
        numbers = re.findall(r'[-+]?\d*\.?\d+', solution)
        if numbers:
            return numbers[-1]
        
        return None
    
    def math_reward_fn(problem: str, solution: str) -> float:
        """Evaluate mathematical reasoning quality."""
        score = 0.0
        
        # Basic checks
        if not solution or len(solution.strip()) < 10:
            return 0.0
        
        # Extract expected answer from problem if available
        import re
        expected_match = re.search(r'Answer:\s*([-+]?\d*\.?\d+)', problem)
        
        # Structural quality (30%)
        structure_score = 0.0
        
        # Check for step-by-step reasoning
        if any(marker in solution.lower() for marker in ['step', 'first', 'then', 'next', 'finally']):
            structure_score += 0.4
        
        # Check for mathematical notation
        math_symbols = ['=', '+', '-', '*', '/', '^', '(', ')']
        symbol_count = sum(1 for symbol in math_symbols if symbol in solution)
        structure_score += min(symbol_count * 0.1, 0.3)
        
        # Check for explanatory text
        explanation_words = ['because', 'since', 'therefore', 'thus', 'so', 'we get', 'this gives']
        explanation_count = sum(1 for word in explanation_words if word in solution.lower())
        structure_score += min(explanation_count * 0.1, 0.3)
        
        score += structure_score * 0.3
        
        # Reasoning quality (40%)
        reasoning_score = 0.0
        
        # Check for logical flow
        solution_lower = solution.lower()
        if 'given' in solution_lower or 'we know' in solution_lower:
            reasoning_score += 0.2
        
        # Check for intermediate steps
        if solution.count('=') >= 2:  # Multiple equations suggest step-by-step work
            reasoning_score += 0.3
        
        # Check for verification or checking
        if any(word in solution_lower for word in ['check', 'verify', 'confirm', 'proof']):
            reasoning_score += 0.2
        
        # Check for proper mathematical language
        math_terms = ['equation', 'solve', 'substitute', 'simplify', 'factor', 'distribute']
        term_count = sum(1 for term in math_terms if term in solution_lower)
        reasoning_score += min(term_count * 0.1, 0.3)
        
        score += reasoning_score * 0.4
        
        # Answer correctness (30%) - if we can verify
        if expected_match:
            expected_answer = expected_match.group(1)
            extracted_answer = extract_answer(solution)
            
            if extracted_answer:
                try:
                    expected_float = float(expected_answer)
                    extracted_float = float(extracted_answer)
                    
                    # Check if answers match (with small tolerance for floating point)
                    if abs(expected_float - extracted_float) < 0.01:
                        score += 0.3
                    elif abs(expected_float - extracted_float) < abs(expected_float) * 0.1:
                        # Within 10% - partial credit
                        score += 0.15
                except ValueError:
                    pass
        else:
            # No expected answer - give benefit of doubt if solution looks complete
            if extract_answer(solution) is not None:
                score += 0.15
        
        # Length appropriateness
        word_count = len(solution.split())
        if 20 < word_count < 300:
            length_bonus = 0.1
        elif 10 < word_count <= 20:
            length_bonus = 0.05
        else:
            length_bonus = 0
        
        score = min(1.0, score + length_bonus)
        
        # Penalties
        if 'error' in solution_lower or 'mistake' in solution_lower:
            score *= 0.7
        
        if solution.count('...') > 2:  # Too many ellipses suggest incomplete work
            score *= 0.8
        
        return max(0.0, score)
    
    return math_reward_fn


def generate_training_problems():
    """Generate diverse mathematical problems for training."""
    problems = []
    
    # Basic arithmetic
    problems.extend([
        "What is 15% of 80? Answer: 12",
        "Calculate 25% of 240. Answer: 60",
        "Find 30% of 150. Answer: 45",
        "What is 12.5% of 96? Answer: 12",
        "Calculate 45% of 200. Answer: 90"
    ])
    
    # Algebra
    problems.extend([
        "Solve for x: 2x + 5 = 13. Answer: 4",
        "Solve for y: 3y - 7 = 14. Answer: 7",
        "Find x: 5x + 2 = 3x + 10. Answer: 4",
        "Solve: 4(x - 3) = 20. Answer: 8",
        "Find x: x/3 + 4 = 7. Answer: 9"
    ])
    
    # Word problems
    problems.extend([
        "A train travels 180 miles in 3 hours. What is its average speed? Answer: 60",
        "If 5 apples cost $3.50, how much do 8 apples cost? Answer: 5.60",
        "A rectangle has length 12 and width 5. What is its perimeter? Answer: 34",
        "If a shirt is discounted 20% from $40, what is the sale price? Answer: 32",
        "A car travels 240 miles using 8 gallons of gas. What is its fuel efficiency in mpg? Answer: 30"
    ])
    
    # Sequences and series
    problems.extend([
        "What is the sum of the first 10 positive integers? Answer: 55",
        "Find the 7th term in the sequence: 2, 5, 8, 11, ... Answer: 20",
        "What is the sum of the first 5 odd numbers? Answer: 25",
        "Find the next number: 1, 4, 9, 16, ... Answer: 25",
        "Calculate the sum: 1 + 2 + 3 + ... + 20. Answer: 210"
    ])
    
    # Geometry
    problems.extend([
        "A circle has radius 7. What is its circumference? (Use π ≈ 3.14) Answer: 43.96",
        "Find the area of a triangle with base 10 and height 6. Answer: 30",
        "A square has perimeter 32. What is its area? Answer: 64",
        "Find the volume of a cube with edge length 5. Answer: 125",
        "A cylinder has radius 3 and height 10. What is its volume? (Use π ≈ 3.14) Answer: 282.6"
    ])
    
    return problems


def generate_test_problems():
    """Generate test problems (without answers for proper evaluation)."""
    return [
        "What is 18% of 75?",
        "Solve for x: 3x - 4 = 17",
        "A car travels 150 miles in 2.5 hours. What is its average speed?",
        "Find the area of a rectangle with length 15 and width 8",
        "What is the sum of the first 8 positive integers?",
        "If a jacket costs $80 after a 20% discount, what was the original price?",
        "Find the next term: 3, 7, 11, 15, ...",
        "A circle has diameter 14. What is its area? (Use π ≈ 3.14)",
        "Solve: 2(3x - 5) = 16",
        "If 12 pens cost $8.40, how much do 15 pens cost?"
    ]


def main():
    """Run the complete reasoning pipeline example."""
    print("=" * 80)
    print("MATHEMATICAL REASONING MODEL DEVELOPMENT")
    print("=" * 80)
    
    # Create reward function
    reward_fn = create_math_reward_function()
    
    # Generate problems
    train_problems = generate_training_problems()
    test_problems = generate_test_problems()
    
    print(f"\nGenerated {len(train_problems)} training problems")
    print(f"Generated {len(test_problems)} test problems")
    
    # Configure pipeline
    config = ReasoningPipelineConfig(
        base_model_name="EleutherAI/gpt-neo-125M",  # Small model for demo
        enable_pure_rl=True,
        enable_cot_training=True,
        enable_rejection_sampling=True,
        enable_prm_training=True,
        enable_distillation=True,
        pure_rl_iterations=50,      # Reduced for demo
        cot_iterations=30,          # Reduced for demo
        rejection_samples=8,        # Reduced for demo
        prm_epochs=2,              # Reduced for demo
        output_dir="./math_reasoning_output"
    )
    
    # Create pipeline
    pipeline = ReasoningModelPipeline(config, reward_fn)
    
    # Run pipeline
    print("\nStarting pipeline...")
    print("Note: This demo uses reduced iterations. For production, use more iterations.")
    
    results = pipeline.run_pipeline(
        train_problems=train_problems,
        test_problems=test_problems,
        student_model_name="microsoft/DialoGPT-small"  # Small student model
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    
    print("\nStages completed:", results['stages_completed'])
    print("\nFinal metrics:")
    for stage, metrics in results['metrics'].items():
        print(f"\n{stage}:")
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {metrics}")
    
    # Test the final model
    print("\n" + "=" * 80)
    print("TESTING FINAL MODEL")
    print("=" * 80)
    
    # Load best model (usually CoT model)
    if 'cot' in results['stages_completed']:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        model_path = os.path.join(config.output_dir, "cot_checkpoint")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Generate some example solutions
        print("\nExample solutions from trained model:")
        for i, problem in enumerate(test_problems[:3]):
            print(f"\nProblem {i+1}: {problem}")
            
            inputs = tokenizer(problem, return_tensors='pt')
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id
                )
            
            solution = tokenizer.decode(outputs[0], skip_special_tokens=True)
            solution = solution[len(problem):].strip()
            
            print(f"Solution: {solution}")
            score = reward_fn(problem, solution)
            print(f"Reward Score: {score:.3f}")
    
    print("\n" + "=" * 80)
    print("Demo complete! Check './math_reasoning_output' for all outputs.")
    print("=" * 80)


if __name__ == "__main__":
    import torch
    
    # Note: This is a demonstration script
    print("This is a demonstration of the reasoning model pipeline.")
    print("For real training, you would need:")
    print("- Larger models (e.g., LLaMA, GPT-3 scale)")
    print("- More training iterations (1000s)")
    print("- More diverse problems (1000s)")
    print("- Better compute resources (GPUs)")
    print()
    
    response = input("Run demonstration? (y/n): ")
    if response.lower() == 'y':
        main()
    else:
        print("Demonstration cancelled.")