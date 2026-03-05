# Evaluation Tools

Evaluation tools provide model assessment and analysis capabilities.

## Available Evaluators

### DQNEvaluator
Evaluates DQN model performance on I-value prediction tasks.

**Key Features:**
- I-value prediction accuracy
- Q-value distribution analysis
- Training progress tracking

## Evaluation Metrics

### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **AUROC**: Area under ROC curve

### Bias Metrics
- **Demographic Performance**: Accuracy per demographic group
- **Bias Score**: Measure of performance disparity across groups
- **Fairness Metrics**: Various fairness definitions

### I-Value Metrics
- **I-Value Distribution**: Distribution of predicted I-values
- **I-Value Accuracy**: Correlation with actual model improvement
- **Exploration Efficiency**: How well I-values guide exploration

## Usage Example

```python
from evaluation.DQNEvaluator import DQNEvaluator

# Initialize evaluator
evaluator = DQNEvaluator(dqn_model)

# Evaluate on test set
results = evaluator.evaluate(test_nodes)

# Access metrics
accuracy = results['accuracy']
f1 = results['f1']
auroc = results['auroc']
```

## Evaluation Process

1. **Load Test Data**: Load nodes from test split
2. **Run Inference**: Get predictions from model
3. **Compute Metrics**: Calculate accuracy, F1, AUROC
4. **Bias Analysis**: Compute per-demographic metrics
5. **Report Results**: Return metrics dictionary

## Notes

- Evaluation is typically done on validation/test sets
- Metrics are computed per demographic group
- Results can be logged to files or visualization tools
