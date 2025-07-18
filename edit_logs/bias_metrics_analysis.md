# Bias Metrics Analysis - Quanty Development Notes

## Current Implementation Analysis

### Bias Metrics Structure
The current bias tracking system in `test_hierarchical.py` uses the following approach:

1. **Categorical Attributes**: 
   - Ground Truth Gender (values: 0, 1)
   - Ground Truth Race (values: 0, 1, 2, 3) 
   - Ground Truth Age (values: 0, 1, 2, 3)

2. **Subgroup Construction**: 
   - Currently creates subgroups using ALL categorical attributes
   - Subgroup keys like: "Ground Truth Gender_0_Ground Truth Race_1_Ground Truth Age_2"
   - This creates very fine-grained subgroups (2 x 4 x 4 = 32 possible combinations)

3. **Bias Metrics Calculated**:
   - Subgroup accuracies for each race-gender-age combination
   - Overall bias (max accuracy difference across all subgroups)
   - Average subgroup bias (average absolute difference from overall accuracy)
   - Per-attribute bias (max accuracy difference within each attribute)
   - Average attribute bias

### Key Code Locations
- Lines 200-220: Subgroup key construction in `evaluate_model()`
- Lines 265-340: Bias metrics calculation and reporting
- Lines 475-520: Attribute metadata definition

## Requested Change
Update subgroup-based bias tracking to use **race-gender subgroups only**, excluding age from the subgroup definition. This will:
- Reduce subgroup granularity from 32 to 8 combinations (2 x 4 = 8)
- Focus bias analysis on race-gender intersections
- Potentially increase sample sizes per subgroup for more reliable metrics

## Implementation Strategy
1. Modify subgroup key construction to filter out 'Ground Truth Age' from categorical attributes used for subgrouping
2. Keep age as categorical attribute for per-attribute bias analysis
3. Update subgroup naming and reporting to reflect race-gender focus
4. Maintain backward compatibility with existing bias metrics structure

## Implementation Details (COMPLETED)

### Changes Made:

1. **Subgroup Construction (Lines ~200-220)**:
   - Added filtering to only use race and gender attributes for subgroups:
   ```python
   race_gender_attrs = [attr for attr in categorical_attrs 
                       if attr['name'] in ['Ground Truth Gender', 'Ground Truth Race']]
   ```
   - Now creates 8 subgroups instead of 32 (2 genders x 4 races = 8 combinations)

2. **Bias Reporting Updates**:
   - Changed print statements to reflect "Race-Gender Subgroup Bias Analysis"
   - Updated bias metrics keys to be more descriptive:
     - `race_gender_subgroup_accuracies`
     - `race_gender_overall_bias` 
     - `race_gender_average_subgroup_bias`

3. **Preserved Functionality**:
   - Age is still included in per-attribute bias analysis
   - All existing bias metrics calculation logic remains intact
   - Backward compatibility maintained with bias metrics structure

### New Subgroup Structure:
- Ground Truth Gender_0_Ground Truth Race_0
- Ground Truth Gender_0_Ground Truth Race_1  
- Ground Truth Gender_0_Ground Truth Race_2
- Ground Truth Gender_0_Ground Truth Race_3
- Ground Truth Gender_1_Ground Truth Race_0
- Ground Truth Gender_1_Ground Truth Race_1
- Ground Truth Gender_1_Ground Truth Race_2
- Ground Truth Gender_1_Ground Truth Race_3

### Benefits:
- Larger sample sizes per subgroup for more reliable bias metrics
- Focused analysis on race-gender intersections
- Clearer bias reporting and interpretation
- Reduced computational complexity for subgroup analysis 

## NEW: Bias Metrics Graphing Implementation

### Requirements
Create comprehensive visualization system for bias metrics tracking over training epochs:

1. **BiasMetricsTracker Class**:
   - Track bias metrics over time (train/val/test)
   - Store epoch-level bias statistics 
   - Generate comprehensive bias visualization plots
   - Save bias data for analysis

2. **Integration Points**:
   - Modify evaluate_model() to return bias metrics in consistent format
   - Add bias tracking to main training loop
   - Create bias plots alongside existing I-value visualizations

3. **Visualization Types**:
   - Race-Gender Subgroup Accuracy Evolution
   - Overall Bias Trend (max accuracy difference)
   - Average Subgroup Bias Trend  
   - Per-Attribute Bias Evolution
   - Bias Heatmaps by epoch
   - Subgroup Performance Comparison

4. **Implementation Plan**:
   - Create trainers/BiasMetricsTracker.py
   - Modify test_hierarchical.py to integrate bias tracking
   - Add bias plotting calls to visualization section
   - Update vault with implementation notes

### Technical Details
- Use matplotlib/seaborn for plots
- Store bias metrics in structured format for JSON serialization
- Ensure compatibility with existing visualization system
- Handle missing data gracefully
- Create publication-ready plots with proper labels/legends

## IMPLEMENTATION COMPLETED ✅

### Files Created/Modified:

1. **NEW: `trainers/BiasMetricsTracker.py`** - Complete bias metrics tracking and visualization system
   - Tracks bias metrics over time for train/val/test splits
   - Generates comprehensive bias visualization plots:
     - Race-Gender Overall Bias Evolution
     - Average Subgroup Bias Trends  
     - Per-Attribute Bias Evolution
     - Accuracy vs Bias Trade-off plots
     - Subgroup Accuracy Heatmaps (latest epoch & evolution)
     - Attribute Bias Comparison charts
   - Saves bias data in JSON format with fallback to pickle
   - Generates comprehensive bias summary reports

2. **MODIFIED: `test_hierarchical.py`** - Integrated bias tracking into training loop
   - Added BiasMetricsTracker import
   - Initialized bias tracker for each test configuration
   - Added training bias evaluation every 5 epochs
   - Added validation bias logging each epoch
   - Added test bias logging for final results
   - Added bias visualization generation after training

### Integration Points:

1. **Initialization** (lines ~920):
   ```python
   bias_save_dir = f"bias_visualizations/{config['description']}"
   bias_tracker = BiasMetricsTracker(save_dir=bias_save_dir)
   ```

2. **Training Bias Evaluation** (lines ~975):
   - Evaluates training bias every 5 epochs and last epoch
   - Uses sampled training nodes for efficiency

3. **Validation Bias Logging** (lines ~1005):
   ```python
   bias_tracker.log_bias_metrics(epoch=epoch, train_metrics=train_metrics_full, val_metrics=val_metrics)
   ```

4. **Test Bias Logging** (lines ~1045):
   ```python
   bias_tracker.log_bias_metrics(epoch=best_epoch-1, test_metrics=test_metrics)
   ```

5. **Visualization Generation** (lines ~1080):
   ```python
   bias_tracker.generate_all_plots()
   ```

### Features Implemented:

1. **Automatic Plot Generation**:
   - 4-panel bias evolution plot (overall bias, avg subgroup bias, attribute bias, accuracy vs bias)
   - Subgroup accuracy heatmaps (latest epoch + evolution over time)
   - Per-attribute bias comparison across splits
   - Bias summary comparison charts

2. **Data Management**:
   - JSON serialization with pickle fallback
   - Timestamp tracking for all metrics
   - Comprehensive metadata storage

3. **Reporting**:
   - Console summary reports with emojis for readability
   - Bias trend analysis (increase/decrease/stable)
   - Best bias-accuracy trade-off identification
   - Split-wise performance comparison

4. **Error Handling**:
   - Graceful handling of missing data
   - Try-catch blocks for visualization generation
   - Fallback data saving mechanisms

### Benefits:
- Comprehensive bias monitoring throughout training
- Publication-ready visualizations
- Automated bias trend analysis
- Integration with existing visualization system
- Minimal performance impact (training bias eval only every 5 epochs)
- Saves all data for post-hoc analysis

### Usage:
The bias tracking is now automatically enabled for all training runs. Visualizations are saved to:
`bias_visualizations/{config_description}/`

No additional command line arguments needed - works out of the box with existing test configurations. 