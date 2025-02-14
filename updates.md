# Updates

## 2025-01-24: Enhanced Graph Node Connection Logic

### Changes to ClusteredDeepfakeDataloader
- Implemented weighted combination of face similarity and attribute matching for node connections
- Added comprehensive attribute importance weights for all facial features:
  - Demographics (Gender, Race, Age)
  - Image Quality (blur, brightness, contrast, compression)
  - Face Alignment (yaw, pitch)
  - Facial Symmetry (overall, eyes, mouth, brows)
  - Emotions (all emotion probabilities)
  - Facial Attributes (hair, eyes, facial hair, accessories)
- Improved attribute similarity calculation with type-specific comparisons:
  - Angle-based similarity for face orientation
  - Continuous similarity for emotions and quality metrics
  - Normalized comparisons for symmetry scores
  - Threshold-based matching for numerical attributes
- Added configurable weights for face similarity vs attribute matching
- Store similarity scores in edge attributes for later analysis

## 2025-01-24: Optimized GPU Acceleration for Face Analysis

### Changes to additional_attributes.py
- Fixed FER initialization to use MTCNN without device parameter
- Optimized GPU usage:
  - DLIB face detection and landmark prediction using CUDA
  - FaceNet model (InceptionResnetV1) and MTCNN on GPU
  - Improved preprocessing for emotion detection
  - DeepFace using internal GPU support
- Restructured code for better parallel processing
- Added better error handling for emotion detection
- Fixed model file paths

## 2025-01-28: Improved Image Processing Performance and Stability

### Changes to additional_attributes.py
- Implemented batch processing to handle large image sets efficiently
- Added ThreadPoolExecutor for parallel processing within batches
- Improved error handling and logging:
  - Added detailed logging for each processing step
  - Better error messages and stack traces
  - Progress tracking for batch operations
- Memory optimizations:
  - Added delays between batches to prevent GPU overload
  - Explicit cleanup of GPU tensors
  - Proper error handling with resource cleanup
- Restructured image processing pipeline:
  - Separate functions for batch and single image processing
  - Better organization of face detection and attribute extraction
  - More robust error handling for missing faces or failed operations

## 2025-01-28: Fixed DataFrame Creation and Memory Issues

### Changes to additional_attributes.py
- Fixed DataFrame creation process:
  - More efficient handling of large result sets
  - Better memory management for face embeddings
  - Improved error handling and logging
- Improved path handling:
  - Fixed image path construction using os.path.join
  - Better validation of image paths
  - Proper ID extraction from paths
- Added incremental processing:
  - Results are processed into records incrementally
  - Separate handling for embeddings to reduce memory usage
  - Better error tracking and reporting

## 2025-01-28: Implemented Chunked Processing for Large Datasets

### Changes to additional_attributes.py
- Added chunked processing to handle large datasets:
  - Process images in chunks of 1000
  - Save results incrementally to disk
  - Combine chunks at the end
- Memory optimizations:
  - Reduced batch size to 16 images
  - Added forced memory cleanup between chunks
  - Implemented proper cleanup of intermediate results
- Added robust error handling:
  - Per-chunk error handling
  - Cleanup of temporary files
  - Better progress tracking and logging
- Improved DataFrame handling:
  - Incremental DataFrame creation
  - Efficient chunk combination
  - Proper memory management for large datasets

## 2025-01-28: Implemented Ultra-Conservative Processing for Large Datasets

### Changes to additional_attributes.py
- Implemented ultra-conservative processing approach:
  - Reduced chunk size to 100 images (from 1000)
  - Reduced batch size to 4 images (from 16)
  - Process single images with memory checks
- Added memory management features:
  - Pre-allocation memory checks
  - Recovery delays for low memory
  - Per-image GPU memory clearing
- Enhanced intermediate saving:
  - Save results every 20 batches
  - Better temporary file handling
  - Improved chunk combination
- Improved error handling:
  - Per-image error catching
  - Chunk-level error recovery
  - Better logging of memory issues

## 2025-01-28: Added Robust Image Validation

### Changes to additional_attributes.py
- Added comprehensive image validation:
  - Pre-processing checks with PIL and OpenCV
  - Image dimension validation
  - Color space verification
  - Multi-step loading process
- Improved error handling:
  - Specific error messages for each failure mode
  - Better logging of validation failures
  - Graceful handling of corrupted images
- Enhanced image processing:
  - Safer color space conversion
  - Better face detection error handling
  - More robust attribute extraction
  - Cleaner error propagation

## 2025-01-28: Added Default Values for Failed Attributes

### Changes to additional_attributes.py
- Added default value handling:
  - Return -1 for numeric attributes on failure
  - Return 'unknown' for categorical attributes
  - Zero vector for face embeddings
  - Default scores for emotions and race
- Improved error tracking:
  - Added detailed error messages
  - Maintain error context through processing
  - Better error propagation
- Enhanced attribute extraction:
  - Independent error handling for each attribute
  - Graceful degradation on partial failures
  - Consistent default values across all failure modes

## 2025-01-28: Added Comprehensive Face Quality Metrics

### Changes to additional_attributes.py
- Added comprehensive quality metrics:
  - Blur score and compression quality
  - Brightness and contrast measures
  - Face alignment angles (yaw, pitch, roll)
  - Facial symmetry measurements
- Enhanced quality calculations:
  - Independent calculation of each metric
  - Proper error handling for each measure
  - Default values for failed calculations
- Improved symmetry analysis:
  - Eye symmetry ratio
  - Mouth symmetry ratio
  - Nose symmetry ratio
  - Overall symmetry score
- Better organization:
  - Grouped related metrics together
  - Consistent measurement scales
  - Clear metric categorization

## 2025-01-28: Fixed Facial Symmetry Calculations

### Changes to additional_attributes.py
- Fixed symmetry calculation issues:
  - Added point interpolation for mismatched regions
  - Better handling of facial landmark points
  - Improved midline calculation
- Enhanced region handling:
  - Careful definition of facial regions
  - Proper flipping of right-side points
  - Better point pairing for comparison
- Added robustness:
  - Better error handling in calculations
  - Proper type conversion for numpy
  - Score normalization and bounds checking
- Improved accuracy:
  - Better distance calculations
  - More precise region definitions
  - Proper handling of facial landmarks

## 2025-01-28: Added Configurable Attribute Processing

### Changes to additional_attributes.py
- Added configuration system:
  - Top-level config dictionary for all attributes
  - Individual toggles for each attribute type
  - Component-level controls for complex attributes
  - Descriptive documentation for each option
- Enhanced attribute processing:
  - Conditional execution based on config
  - Granular control over components
  - Better resource utilization
- Improved organization:
  - Structured configuration layout
  - Clear default value handling
  - Better code modularity
- Added flexibility:
  - Easy to enable/disable features
  - Fine-grained control over processing
  - Simple configuration updates

## 2025-01-28: Implemented GPU-Accelerated Batch Processing

### Changes to additional_attributes.py
- Added batch processing:
  - Process multiple images in parallel
  - GPU-accelerated face embeddings
  - Configurable batch size
  - Memory-efficient loading
- Improved performance:
  - Reduced GPU idle time
  - Better resource utilization
  - Faster face embedding extraction
- Enhanced robustness:
  - Better error handling per batch
  - Graceful failure recovery
  - Progress tracking per batch
- Memory management:
  - Efficient batch loading
  - Proper GPU memory cleanup
  - Reduced memory footprint

## 2025-01-31: Updated Clustered Deepfake Dataloader Attributes

### Changes to ClusteredDeepfakeDataloader.py
- Added face_roll attribute:
  - Added to attribute importance weights
  - Updated angle similarity calculation to include roll
- Updated symmetry metrics:
  - Renamed symmetry_score to eye_ratio
  - Renamed eye_symmetry to mouth_ratio
  - Renamed mouth_symmetry to nose_ratio
  - Renamed brow_symmetry to overall_symmetry
- Enhanced similarity calculations:
  - Updated attribute matching for new metrics
  - Improved ratio comparison logic
  - Better handling of symmetry scores

## 2025-02-13: Simplified Image Processing Pipeline

### Changes to additional_attributes.py
- Removed chunking system:
  - Eliminated intermediate file creation
  - Removed chunk-based processing
  - Simplified file handling
- Enhanced batch processing:
  - Single-pass batch processing
  - Direct DataFrame output
  - Cleaner memory management
- Improved CLI:
  - Updated argument names
  - Better parameter descriptions
  - Simplified workflow
- Memory optimization:
  - Removed redundant memory cleanup
  - Streamlined data flow
  - Reduced disk I/O

## 2025-02-13: Added Debug Visualization Feature

### Changes to additional_attributes.py
- Added debug visualization:
  - Creates labeled copies of processed images
  - Shows quality metrics, alignment, symmetry, and emotions
  - Semi-transparent text overlays with background
  - Auto-scaling text based on image size
- Enhanced visualization:
  - Section-based organization
  - Clear formatting and spacing
  - High contrast text with backgrounds
  - Proper value formatting
- Added CLI options:
  - --debug_vis flag to enable visualization
  - --debug_vis_dir for output directory
  - --max_debug_images to limit output (default 100)
- Improved usability:
  - Non-destructive (creates copies)
  - Progress tracking
  - Error handling per image

## 2025-02-13: Enhanced Debug Visualization with Landmarks
- Added color-coded facial landmark visualization:
  - Eyes: Green
  - Mouth: Red
  - Nose: Yellow
  - Nose bridge: Cyan
- Added connecting lines between landmarks
- Added point markers for each landmark
- Improved debugging capabilities for symmetry calculation

## 2025-02-13: Enhanced Landmark Visualization
- Improved landmark visualization:
  - Distinct colors for left/right features:
    - Left eye: Green
    - Right eye: Orange
    - Left mouth: Red
    - Right mouth: Purple
    - Left nose: Yellow
    - Right nose: Cyan
    - Nose bridge: White
  - Larger point markers (3px)
  - Thicker connecting lines (2px)
  - Better visibility against dark backgrounds

## 2025-02-13: Fixed Debug Visualization Colors
- Fixed color space handling in debug visualization:
  - Removed redundant BGR conversion
  - Fixed blue tint in output images
  - Images now display with correct colors

## 2025-02-13: Fixed Nose Symmetry Calculation
- Improved nose symmetry calculation:
  - Expanded nose landmark points to include nostrils
  - Fixed incorrect landmark indices
  - More accurate nose symmetry measurements

## 2025-02-13: Improved Symmetry Calculation
- Enhanced nose symmetry calculation:
  - Fixed point order for right nostril comparison
  - Added proper flipping of right nostril points
  - Added division by zero protection
- Added detailed debug logging:
  - Point counts and interpolation
  - Distance calculations
  - Raw symmetry scores
  - Per-feature symmetry scores
- Improved error handling:
  - Better error messages
  - Graceful fallback for face detection
  - More informative warnings

## 2025-02-13: Fixed Symmetry Calculations
- Fixed mouth region definition:
  - Now properly split into left (48-52) and right (52-55) sides
  - Previously used top and bottom incorrectly
- Enhanced nose symmetry debugging:
  - Added detailed point coordinate logging
  - Added distance calculation debugging
  - Added midline vector validation
  - Added interpolation verification
- Improved numerical stability:
  - Added midline length check
  - Better handling of zero distances
  - More robust cross product calculation

## 2025-02-13: Enhanced Symmetry Scoring
- Improved symmetry score calculation:
  - Now preserves negative scores for asymmetric features
  - 1.0 = perfect symmetry (distances match exactly)
  - 0.0 = neutral symmetry (differences equal mean distance)
  - negative = asymmetric (differences larger than mean distance)
- Added detailed docstrings explaining scoring system
- Removed score clamping to preserve more information

## 2025-02-13: Fixed Distance Calculation
- Fixed perpendicular distance calculation:
  - Now properly normalizes midline vector
  - Uses dot product with perpendicular vector instead of cross product
  - More accurate distance measurements for symmetry calculation
- Added more detailed debug logging for distance calculations

## 2025-02-13: Improved Logging System
- Added global debug mode flag:
  - `enable_debug_logging()` function to enable detailed logging
  - Debug logs written to `logs/symmetry_debug.log`
  - Warning-level logs by default
- Cleaned up logging:
  - Removed console debug output
  - Debug info only logged when debug mode is enabled
  - Only important warnings shown in console
- Added conditional debug info in return values

## 2025-02-13: Improved Progress Bar Display
- Enhanced progress tracking:
  - Shows total batch count (e.g., "0/10 batches")
  - Uses context manager for cleaner progress updates
  - Properly updates progress for skipped batches
  - Maintains consistent position in console

## 2025-02-13: Fixed Progress Bars
- Improved progress bar display:
  - Main batch progress bar stays at top
  - Image processing bar updates below
  - No more duplicate progress bars
  - Cleaner console output

## 2025-02-14: Progress Bar Improvements
- Changed progress bar to track individual images instead of batches
- Added incremental updates for skipped/failed images
- Maintained all existing image processing functionality

## 2025-02-14: Performance Optimization for Facial Attribute Processing
- Implemented multiprocessing for CPU-intensive tasks using Python's multiprocessing pool
- Optimized batch processing for emotion detection and face embeddings
- Improved memory management with periodic cleanup and explicit garbage collection
- Added better error handling and isolation
- These changes should significantly reduce processing time for large datasets

## 2025-02-14: Hierarchical Node Matching System

Added a new hierarchical node matching system to the `UnclusteredDeepfakeDataloader` class that:
1. Organizes attributes into categories (race, gender, age, emotion, facial features, quality metrics)
2. Performs hierarchical matching by filtering node pairs based on attribute similarity in order
3. Handles both categorical and numerical attributes with appropriate similarity metrics
4. Replaces the previous LSH-based matching system with a more precise attribute-based approach

Changes made:
- Updated `_create_attribute_matrix()` to handle categorized attributes and numerical values
- Added `_compute_similarity()` for attribute-specific similarity calculations
- Added `_hierarchical_match()` for the new matching algorithm
- Modified `process_node_batch()` to use the hierarchical matching system
- Removed LSH-based matching code as it's no longer needed

The new system ensures more accurate matches by:
1. First matching faces with the same race
2. Then filtering by gender
3. Then filtering by age similarity
4. Then filtering by emotional expression
5. Then considering facial features
6. Finally considering quality metrics

This creates a more natural grouping of faces that share fundamental characteristics.

## 2025-02-14: Updated Similarity Computation

Modified the similarity computation in the hierarchical node matching system:

1. Categorical Features (race, gender, age, emotion, facial features):
- Changed to use exact matching instead of cosine similarity
- Two nodes match only if they have identical values for the category
- No partial matches or similarity scores - it's binary match/no-match

2. Numerical Features (quality metrics):
- Added specific acceptable ranges for each metric:
  - Symmetry: ±0.3 difference allowed
  - Blur: ±50 difference allowed
  - Brightness: ±50 difference allowed
  - Contrast: ±50 difference allowed
  - Compression: ±20 difference allowed
- Computes match ratio based on how many metrics are within acceptable range
- Only compares metrics that are present in both nodes
- Removed problematic normalization that didn't handle large or negative values well

These changes make the matching system more precise for categorical features while providing appropriate tolerance ranges for numerical metrics.

## 2025-02-14: Added Face Embeddings Comparison

Updated the node matching system to include face embeddings comparison:

1. Feature Matrix Changes:
- Removed 'facial_features' category as it's replaced by face embeddings
- Added separate embeddings matrix to store 512-dimensional FaceNet embeddings
- Updated attribute collection to handle both regular attributes and embeddings

2. Similarity Computation:
- Added dedicated embeddings comparison using cosine similarity
- Embeddings comparison threshold set to 0.7 (adjustable)
- Kept exact matching for categorical features
- Maintained specific ranges for quality metrics

3. Hierarchical Matching Order:
- Updated matching order: race → gender → age → emotion → embeddings → quality metrics
- Face embeddings comparison happens after basic demographic matching
- Only pairs that match on demographics and have similar embeddings proceed to quality metrics comparison

This change provides a more accurate way to compare facial features using the deep learning embeddings instead of individual feature matching.

## 2025-02-14: Implemented DQN-based I-Value Predictor

### Changes to DQNIValuePredictor.py
- Implemented neural network-based predictor for node rewards and Q-values
- Added experience replay memory with automatic forgetting of old traces
- Created efficient GPU-accelerated training pipeline
- Implemented I-value prediction based on normalized Q-values
- Added separate reward prediction functionality
- Improved memory management with batch processing
- Added automatic model initialization based on node attributes

## 2025-02-14: Updated AIFaceDataset with Additional Attributes

Enhanced the AIFaceDataset to load and incorporate additional attributes from CSV files:

1. Added Attribute Loading:
- Created `_load_additional_attributes()` method to load attributes from CSV files
- Supports train_attributes.csv, val_attributes.csv, and test_attributes.csv
- Files are expected in the same directory as the dataset
- Handles missing files and invalid data gracefully

2. Node Creation Updates:
- Modified `create_node()` to accept additional attributes
- Base attributes (Gender, Race, Age) are preserved
- Additional attributes are merged with base attributes
- Attributes are matched to nodes using filenames

3. Implementation Details:
- Efficient CSV reading with pandas
- Automatic NaN value filtering
- Memory-efficient processing with chunking
- Maintains existing multiprocessing capabilities
- Added informative logging for attribute loading status

This update enables the dataset to incorporate new attributes while maintaining the existing functionality and performance optimizations.

## 2025-02-14: Added Performance-Based Graph Manager

Added a new `PerformanceGraphManager` that dynamically rewires the graph based on model performance:

- Created `managers/PerformanceGraphManager.py` implementing dynamic graph rewiring
- Uses I-value predictions to identify weak and strong nodes
- Automatically adds edges to nodes with poor performance
- Removes edges from nodes with consistently good performance
- Updated test.py to use the new graph manager

The manager helps focus computational resources on difficult examples while reducing overhead on well-learned examples.

## 2025-02-14: Integrated Performance Tracking with IValueTrainer

Enhanced the integration between `PerformanceGraphManager` and `IValueTrainer`:

- Connected DQN models from IValueTrainer to graph manager for I-value prediction
- Added performance tracking in IValueTrainer's node processing pipeline
- Automatic graph updates now occur during training based on node performance
- Streamlined traversal creation and trainer initialization in test.py
- Fixed missing graph manager updates in the training loop

These changes ensure that the graph structure is properly rewired based on model performance during training.

## 2025-02-14: Restored and Improved Traversal Step Settings

Enhanced traversal configuration in test.py:

- Restored step settings for all traversals
- Training uses fixed 2000 steps for consistent training size
- Validation and test use exact dataset sizes to ensure complete coverage
- Added informative logging of graph sizes and traversal settings
- Maintained single pointer configuration for all traversals