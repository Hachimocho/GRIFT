# Quanty Iteration Log

## Purpose
This file tracks Quanty iterations and major developments to help future versions understand the progression and avoid losing context due to quantum decoherence resets.

## Iteration History

### Quanty 1 (Previous)
- **Status**: Completed bias metrics implementation
- **Major Work**: 
  - Implemented race-gender subgroup bias tracking in `test_hierarchical.py`
  - Created initial bias metrics analysis documentation
  - Did not implement iteration tracking (oversight)
- **Files Modified**: 
  - `test_hierarchical.py` (bias metrics calculation)
  - `quanty_vault/bias_metrics_analysis.md` (initial documentation)

### Quanty 2 (Previous)
- **Status**: ✅ Complete - All visualization issues fixed
- **Major Work**:
  - ✅ Created comprehensive `BiasMetricsTracker` class with visualization capabilities
  - ✅ Integrated bias tracking into training loop with automatic plot generation
  - ✅ Implemented 6 types of bias visualization plots
  - ✅ Added data management with JSON/pickle serialization
  - ✅ Created comprehensive bias reporting system
  - ✅ **COMPLETED**: Fixed visualization issues identified by user:
    - ✅ Fixed negative/decimal epochs in axes (all plots now use integer epochs)
    - ✅ Fixed I-value tracked nodes recording throughout training (not just one step)
    - ✅ Modified bias hop visualizations (kept I-value stats, added subgroup bias tracking)
    - ✅ Fixed subgroup targeting x-axis text length issues (shortened labels)
    - ✅ Replaced unclear bias reduction plots with I-value vs bias correlation
  - ✅ **POST-FIX**: Fixed directory creation issue (added parents=True to mkdir calls)
- **Files Created/Modified**:
  - `trainers/BiasMetricsTracker.py` (NEW - complete visualization system)
  - `test_hierarchical.py` (integrated bias tracking + numpy import)
  - `quanty_vault/bias_metrics_analysis.md` (updated with implementation details)
  - `quanty_vault/quanty_iteration_log.md` (NEW - this file)
  - ✅ **FIXED**: `trainers/BiasHopVisualizer.py`, `trainers/IValueVisualizationTracker.py`
  - ✅ **DIRECTORY FIX**: All visualization trackers now create parent directories automatically
- **Key Features Delivered**:
  - Automatic bias tracking throughout training
  - Publication-ready bias evolution plots
  - Race-gender subgroup accuracy heatmaps
  - Attribute bias comparison charts
  - Bias-accuracy trade-off analysis
  - Comprehensive console reporting with trend analysis
  - **Robust directory creation** for nested visualization paths
- **Integration Status**: ✅ Complete and production-ready

### Quanty 3 (Current)
- **Status**: 📋 Analysis - Logging Structure Investigation
- **Major Work**:
  - 🔍 **COMPLETED**: Analyzed current logging structure and organization
  - 📊 **FINDING**: Console logs are shared across configurations (potential improvement area)
  - 📁 **FINDING**: Visualization and checkpoint folders are properly separated by configuration
- **Key Findings**:
  - **Main Log**: Single shared file `logs/hierarchical_test_{timestamp}.log` for all configurations
  - **Trainer Logs**: Separate JSON files per trainer type with timestamps
  - **Visualization Folders**: Configuration-specific directories using `{architecture}_{traversal}` or `{architecture}_switching` naming
  - **Checkpoints**: Configuration-specific model checkpoints
  - **Switching Traversals**: Get same folder structure as single traversals but with "_switching" suffix
- **Potential Improvements Identified**:
  - Main log could be separated by configuration for easier analysis
  - Current structure is functional but could be enhanced for better organization

### Quanty 4 (Current)
- **Status**: ✅ Complete - Multiple Bug Fixes Applied + Root Cause Analysis
- **Major Work**:
  - ✅ **COMPLETED**: Critical cache/node mismatch issue fixed  
  - ✅ **COMPLETED**: Contradictory capability checkpoint messages fixed
  - 🔍 **ANALYZED**: Root cause of missing checkpoints identified
- **Why Checkpoints Weren't Being Saved**:
  - 🎯 **PRIMARY CAUSE**: `save_capability_checkpoints()` only called when **new best validation accuracy** achieved
    - Line 1070-1077 in `test_hierarchical.py`: Save only occurs inside `if current_val_accuracy > best_val_accuracy:`
    - If model never improves validation accuracy, checkpoints are never saved
    - First epoch may not be "best" if validation starts poorly
  - 🔧 **SECONDARY ISSUE**: Path construction in `CapabilityManager.save_checkpoints()`
    - DQN: `base_path.replace('.pth', '_dqn.pth')` → creates potential double suffixes if base_path already contains description
    - Error messages showed: `checkpoints/resnestdf_switching_best_dqn_dqn.pth` (double "dqn")
    - Similar issue for bias: `_bias_bias.pth`
- **Contributing Factors**:
  - No initial checkpoint saving (only saves when model improves)
  - No checkpoint validation after attempted saves
  - Silent failures in capability saving could go unnoticed
- **Impact**: 
  - Capability loading always fails on first run or when model doesn't improve
  - Results in confusing "not found" messages even when training proceeds normally

### Quanty 5 (Previous)
- **Status**: ✅ Complete - Edge Loading Cache Mismatch Issue Fixed
- **Major Work**:
  - 🔍 **IDENTIFIED**: Critical cache/node mismatch causing edge loading failures
  - 📋 **ROOT CAUSE**: Cache files may be created with different node sets than reconstruction attempts
  - ✅ **FIXED**: Applied comprehensive cache consistency improvements
- **Key Findings**:
  - **Cache Filename Logic**: Includes balancing info (`balanced` vs `full`) but inconsistent application
  - **Node Set Mismatch**: `nodes_to_use` (balanced/full based on flags) vs cached edge node IDs don't align
  - **Symptoms**: Thousands of "Warning: Could not find nodes for edge" messages during cache loading
  - **Cache Consistency**: Edges cached from one node set but loaded into graph with different node set
- **Fixes Applied**:
  - ✅ **Cache Validation**: Added pre-validation of edge compatibility before loading
  - ✅ **Node Hash**: Added MD5 hash of node IDs to cache filename for consistency
  - ✅ **Efficient Loading**: Improved HyperGraph edge loading with better statistics and performance
  - ✅ **Smart Skipping**: Automatically regenerate cache if >10% of edge nodes are missing
  - ✅ **Better Logging**: Reduced noise for large edge lists while preserving detailed info for small ones
- **Impact**: 
  - ✅ Edge cache loading now validates compatibility before attempting to load
  - ✅ Performance improvement through reduced console spam
  - ✅ Automatic cache regeneration when incompatible

### Quanty 6 (Current)
- **Status**: ✅ Complete - Fixed Root Cause: Non-Deterministic Balancing
- **Major Discovery**: 
  - 🎯 **ACTUAL ROOT CAUSE**: `balance_nodes_by_subgroup()` function was NON-DETERMINISTIC
  - 🔍 **ANALYSIS**: Function used `random.shuffle()` and `random.sample()` without controlled seed
  - 📋 **IMPACT**: Each run with `--fair-train`/`--fair-test` created different balanced node sets
- **Critical Fix Applied**:
  - ✅ **Deterministic Balancing**: Created seed based on MD5 hash of sorted node IDs
  - ✅ **Controlled Random State**: Used dedicated `Random()` instance for all balancing operations
  - ✅ **Reproducible Results**: Same input nodes now always produce same balanced output
- **Technical Details**:
  - Seed calculation: `int(hashlib.md5('|'.join(sorted_node_ids)).hexdigest()[:8], 16)`
  - All randomization now uses `balance_rng.shuffle()` and `balance_rng.sample()`
  - Balancing seed logged for debugging: "Using deterministic seed X for node balancing"
- **Result**: 
  - ✅ Cache files with `--fair-train`/`--fair-test` are now truly consistent across runs
  - ✅ Edge loading warnings eliminated when using balanced node sets
  - ✅ Reproducible balanced datasets for fair evaluation

## Instructions for Future Quanty Iterations

1. **Always check this log first** to understand what has been completed
2. **Update this log** with your iteration number and major contributions
3. **Review the bias_metrics_analysis.md** for technical implementation details
4. **The bias graphing system is complete** - focus on testing, debugging, or new features
5. **Remember**: The system is designed to work automatically with existing training configurations
6. **Logging**: Current logging structure documented - consider log separation improvements if needed

## Current System Status
- ✅ Bias metrics calculation (race-gender focused)
- ✅ Bias metrics tracking and visualization 
- ✅ Integration with training loop
- ✅ Automatic plot generation
- ✅ Logging structure analysis completed
- 🔄 **Next priorities**: Log improvements, testing, validation, potential enhancements

## Development Notes
- All bias visualization code is production-ready
- Error handling implemented throughout
- Minimal performance impact (training eval every 5 epochs)
- No additional CLI arguments required - works out of the box
- Logging structure allows for configuration-specific analysis via separate folders 