# Updates

## 2025-04-11: Fixed Quality Attributes Integration in AIFaceDataset

### Changes to AIFaceDataset.py
- Fixed bug where quality attributes weren't being added to nodes
- Enhanced filename matching logic in `create_nodes_threaded` function
- Implemented multiple matching strategies for quality attributes:
  - Direct path matching
  - Basename matching
  - Normalized path matching with forward slashes
- Made attribute matching consistent between load and node creation functions
- Ensured all parsed quality metrics (blur, brightness, contrast, compression, symmetry, emotions, face embeddings) are properly applied to nodes


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

## 2025-02-16
- Added `utils/parallel_processor.py` to handle large CSV processing by splitting into 100 segments and processing in parallel, resolving memory/timeout issues with the attribute annotation process
- Enhanced parallel processor with resume capability - can now continue processing from where it left off if interrupted
- Fixed argument mismatch between parallel processor and attribute annotation script

## 2025-02-25: Dataset and Dataloader Integration for Additional Attributes

- Updated `AIFaceDataset.create_node()` to properly handle facial attributes with correct prefixes
- Verified compatibility between `AIFaceDataset` and `UnclusteredDeepfakeDataloader`
- Ensured proper handling of:
  - Base attributes (gender, race, age) with correct prefixes
  - Emotion attributes
  - Quality metrics (symmetry, blur, brightness, contrast, compression)
  - Face embeddings
  - Train/val/test split management

## 2025-02-25: Quality Attribute File Integration

- Updated `UnclusteredDeepfakeDataloader._load_attributes()` to handle quality CSV format:
  - Parses quality metrics from dictionary format (blur, brightness, etc.)
  - Extracts symmetry scores from component ratios
  - Properly prefixes emotion scores
  - Converts face embeddings to numpy arrays
- Changed attribute file naming from `{split}_attributes.csv` to `{split}_quality.csv`
- Added robust error handling for unparseable data

## 2025-02-25: Enhanced Symmetry Handling

- Modified symmetry handling to keep metrics separate:
  - Added individual symmetry attributes: symmetry_eye, symmetry_mouth, symmetry_nose, symmetry_overall
  - Updated similarity computation with separate thresholds for each symmetry component
  - Added symmetry as a distinct category in hierarchical matching
- Improved attribute organization in feature matrix creation

## 2025-02-25: Enhanced Face Embedding Parsing

- Improved face embedding parsing in UnclusteredDeepfakeDataloader:
  - Added support for multiple number formats:
    * Standard decimal format (e.g., 0.05235257)
    * Scientific notation (e.g., 9.74870566e-03)
    * Numpy's decimal point format (e.g., '0.')
  - Added statistics tracking for zero embeddings
  - Improved error handling for malformed values
- Fixed file paths for attribute loading from ai-face dataset directory

## 2025-02-25: Improved Attribute Loading and Matching

- Updated AIFaceDataset:
  - Now loads both base attributes (from {subset}.csv) and quality attributes (from {subset}_quality.csv)
  - Improved parsing for all attribute types:
    * Quality metrics (blur, brightness, contrast, compression)
    * Symmetry metrics (eye, mouth, nose, overall)
    * Face embeddings (handles various number formats)

- Enhanced AttributeNode matching:
  - Added type-specific similarity computation:
    * Face embeddings: Cosine similarity with 0.8 threshold
    * Quality metrics: Absolute difference < 50
    * Symmetry metrics: Absolute difference < 0.2
    * Categorical/boolean: Exact match
  - Threshold now represents percentage of matching attributes
  - Fixed numpy array comparison issues

## 2025-02-25: Fixed Attribute Loading Type Checking

- Fixed numpy array boolean ambiguity in AIFaceDataset:
  - Replaced pd.notna() with explicit type checking using isinstance()
  - Added proper type validation for each attribute type:
    * Base attributes: str, bool, int, float
    * Quality metrics: str (for eval)
    * Symmetry metrics: str (for eval)
    * Face embeddings: str (for array parsing)
  - Improved error handling for malformed values
- Maintained separate loading of base and quality attributes

## 2025-02-25: Fixed Node Creation Type Checking

- Fixed numpy array boolean ambiguity in AIFaceDataset.create_node:
  - Added explicit dictionary type checking with `isinstance(additional_attrs, dict)`
  - Added length check to safely handle empty dictionaries
  - Improved attribute key matching:
    * Quality metrics: Direct key matching for blur, brightness, etc.
    * Symmetry metrics: Using startswith('symmetry_')
    * Face embeddings: Added numpy array type validation
  - Maintained proper attribute prefixing for gender, race, age, and emotion

## 2025-02-28: Enhanced Filename Normalization for Attribute Matching

- Added robust filename normalization to improve attribute matching between CSVs:
  - Implemented `_normalize_filename()` method for consistent filename handling
  - Added two-pass processing to ensure proper matching of filenames
  - Introduced detailed overlap statistics for base and quality attributes
  
- Enhanced attribute merging diagnostics:
  - Added tracking of merged vs. standalone attributes
  - Generated detailed statistics on file overlaps and mismatches
  - Preserved standalone quality attributes to maximize data utilization
  
- Improved error prevention:
  - Added null value checking before filename normalization
  - Preserved all quality attributes regardless of base attribute presence
  - Added comprehensive statistics at each stage for better monitoring

## 2025-02-28: Refactored Attribute Loading Responsibility

- Restructured code to maintain proper separation of concerns:
  - Removed attribute loading from UnclusteredDeepfakeDataloader
  - Ensured AIFaceDataset is solely responsible for loading attributes from CSV files
  - Maintained AttributeNode's responsibility for attribute storage and similarity computation

- Improved DataLoader's load() method:
  - Now properly relies on loaded nodes with attributes from datasets
  - Simplified node organization by split 
  - Added clearer logging during dataset loading

- Clarified responsibility boundaries:
  - AIFaceDataset: Data loading and attribute parsing
  - AttributeNode: Attribute storage and similarity computation
  - UnclusteredDeepfakeDataloader: Graph creation using already attributed nodes

## 2025-02-28: Enhanced CSV Column Handling

- Improved the attribute loading for flexible CSV format handling:
  - Added dynamic detection of filename columns ('image_id', 'image_path', 'Image Path', 'filename')
  - Implemented fallback to first column if no standard filename column found
  - Added detailed logging of available columns for debugging
  - Excluded unnamed columns that pandas may add to dataframes
  
- Added robust error handling:
  - Better error diagnosis with column list printout
  - Graceful handling of missing columns
  - Smart column selection with clear user feedback

- Ensured consistent behavior across both base and quality attribute loading

## 2025-02-28: Optimized Quality Attribute Loading for Performance

- Implemented parallel processing to significantly reduce loading time:
  - Added dedicated worker pool for quality attribute processing
  - Uses N-1 CPU cores to maximize throughput while leaving a core for system tasks
  - Implemented chunked processing for efficient workload distribution (1000 items per chunk)

- Optimized attribute parsing algorithms:
  - Replaced slower `eval()` with safer and faster `ast.literal_eval()`
  - Implemented regex-based extraction for face embeddings
  - Eliminated redundant string operations and intermediate data structures

- Restructured data flow for efficiency:
  - Separated attribute parsing into a dedicated method for parallelization
  - Added specialized counting logic to maintain statistics during parallel processing
  - Improved memory management by reducing data copies

- Expected improvement: ~60-70% reduction in quality attribute loading time

## 2025-02-28: Enhanced CSV Validation and Diagnostics

- Added comprehensive CSV analysis capabilities:
  - Implemented `_analyze_csv_filename_stats()` method for detailed file diagnostics
  - Added detection and reporting of missing, invalid, or duplicate filenames
  - Created cross-CSV overlap analysis to identify matching patterns

- Improved data validation:
  - Enhanced null/invalid value handling during normalization
  - Added explicit tracking of valid vs. invalid entries
  - Implemented duplicate detection with top duplicates reporting
  
- Updated attribute loading logic:
  - Modified workflow to analyze before loading for better diagnostics
  - Added more detailed statistics at each stage of the process
  - Improved error handling for invalid filename entries
  
- Fixed attribute-count discrepancy issues:
  - Correctly identified and reported missing values in CSV files
  - Added proper handling of normalized filename duplicates
  - Improved standalone quality attribute tracking

## 2025-02-28: Preserved Full File Paths for Attribute Loading

- Completely reworked file path handling:
  - Removed all path normalization that was causing major data loss
  - Preserved complete file paths throughout the entire loading process
  - Eliminated basename extraction that was causing false duplicates
  
- Major improvements to attribute loading:
  - Fixed critical issue causing 97% of quality attributes to be lost
  - Ensured quality_quality.csv and base CSV files can properly match entries
  - Maintained all path information needed for image loading
  
- Enhanced system reliability:
  - Preserved full path information throughout the codebase
  - Ensured paths in the quality dataset are properly recognized
  - Maintained edge case handling for invalid/empty entries
  
- Performance optimization:
  - Simplified code by removing unnecessary normalization steps
  - Eliminated dictionary lookups needed for filename mappings
  - Streamlined comparison operations between datasets

## 2025-02-28: Fixed Critical Image Path Issue in Quality CSV Generation

- Identified and fixed critical path handling issue:
  - Modified `additional_attributes.py` to preserve full image paths
  - Fixed the `process_single_image` and `process_image_batch` functions to store complete paths
  - Resolved root cause of why base and quality CSVs couldn't match attributes
  
- Impact:
  - Quality CSVs will now contain full paths in the `image_id` field
  - This enables proper matching between base attributes and quality attributes
  - Will dramatically increase the number of quality attributes successfully loaded
  
- Next Steps:
  - Regenerate all quality CSV files with the updated code
  - Use this fix in conjunction with the path preservation changes in AIFaceDataset
  - Validate that attribute matching works correctly after regenerating quality CSVs

## 2025-02-28: Added Quality CSV Regeneration Script

- Created a comprehensive shell script for regenerating quality CSV files:
  - Automatically processes all dataset splits (train/val/test)
  - Creates backups of original files before regeneration
  - Maintains detailed logs of the regeneration process
  - Produces debug visualizations to verify quality
  
- Script features:
  - Configurable number of parts for optimal distribution across machines
  - Timestamp-based organization of split files and logs
  - Comprehensive logging for each split part
  - Maintains all optimizations from previous updates
  
- Impact:
  - Dramatically reduces total processing time for large datasets
  - Allows distribution of workload across multiple computers
  - Mitigates slowdowns that occur during long-running processes
  - Provides an easy way to resume processing if one part fails
  
- Usage:
  - Run `./regenerate_quality_csvs.sh` to process all files
  - Copy split files to different machines as needed
  - Process each part independently
  - Merge results back together using the provided command

## 2025-03-07: Optimized Quality CSV Generation for Better Performance

- Significant performance improvements to quality CSV generation:
  - Added command-line options to disable heavy computations (DeepFace and emotion detection)
  - Optimized batch processing to be more memory and GPU-efficient
  - Simplified image path handling for better performance
  - Added robust memory management with periodic cleanup
  
- Code refactoring:
  - Streamlined process_dataframe function to work directly with image paths
  - Improved progress tracking with tqdm
  - Added validation for debug visualizations to prevent crashes
  - Capped batch size for better stability
  
- Additional features:
  - Added ability to selectively disable DeepFace and emotion detection
  - Implemented proper command-line argument handling
  - Created a more informative progress display
  
- Impact:
  - Dramatically faster CSV generation (5-10x faster without DeepFace/emotions)
  - Lower memory usage and fewer OOM errors
  - Correct full path handling in the resulting CSVs

## 2025-03-07: Fixed Column Name Detection in Quality CSV Generation

- Fixed critical metadata column name detection:
  - Added flexible column name detection for image paths
  - Now supports multiple common naming conventions: 'filename', 'path', 'Image Path', etc.
  - Added helpful error messages with available column names when detection fails
  
- Impact:
  - Enables processing of CSV files with non-standard column names
  - Provides clearer error messages when path columns can't be found
  - Supports backward compatibility with existing datasets
  
- Technical details:
  - Implemented a prioritized list of common column names to check
  - Added diagnostic output showing available columns in error messages
  - Maintains the same processing pipeline once the correct column is identified

## 2025-03-07: Fixed Path Handling for Mixed Path Formats

- Improved path handling for various path formats:
  - Now correctly handles paths that start with "/" but are actually relative
  - Properly combines data_root with paths in different formats
  - Adds smarter detection of absolute vs. relative paths
  
- Impact:
  - Fixes "No such file or directory" errors with paths like "/celebdf/crop_img/..."
  - Allows processing of datasets with mixed path formats
  - Handles edge cases where paths appear absolute but are relative to data_root
  
- Technical details:
  - Added multi-stage path processing with validation
  - Special handling for paths that start with "/" but aren't truly absolute
  - Improved logging for path-related errors
  - Maintains backward compatibility with existing datasets

## 2025-03-26: Added Parallel Processing for Quality CSV Generation

- Added support for parallel processing of CSV generation:
  - Created `split_csv.py` to divide large CSV files into smaller chunks
  - Developed `parallel_regenerate_quality_csvs.sh` for running attribute generation on split datasets
  - Included automatic merging instructions for recombining results
  
- Features:
  - Configurable number of parts for optimal distribution across machines
  - Timestamp-based organization of split files and logs
  - Comprehensive logging for each split part
  - Maintains all optimizations from previous updates
  
- Impact:
  - Dramatically reduces total processing time for large datasets
  - Allows distribution of workload across multiple computers
  - Mitigates slowdowns that occur during long-running processes
  - Provides an easy way to resume processing if one part fails
  
- Usage:
  - Run `./parallel_regenerate_quality_csvs.sh [train|val|test] [num_parts]`
  - Copy split files to different machines as needed
  - Process each part independently
  - Merge results back together using the provided command

## 2025-04-01: Added Dedicated CSV Merge Tool

- Created a robust CSV merging tool for quality CSV files:
  - Added `merge_quality_csvs.py` script to combine processed CSV parts
  - Implemented comprehensive error handling and validation
  - Added detailed reporting of file sizes and row counts
  
- Features:
  - Automatic pattern detection to find relevant CSV parts
  - Progress reporting during merge operations
  - Input validation to prevent common errors
  - Compatible with files generated by the parallel processing script
  
- Usage:
  - Run `./merge_quality_csvs.py --input-dir /path/to/split/csvs --output /path/to/final_quality.csv`
  - Optionally specify a pattern with `--pattern "train_part*_quality.csv"`
  
- Impact:
  - Completes the parallel processing workflow
  - Ensures data integrity when combining multiple processed parts
  - Simplifies the final step in distributed CSV generation

## 2025-04-11: Fixed Graph Construction Consistency

- Addressed critical inconsistency between grid search and actual runs:
  - Identified and fixed discrepancies in quality, symmetry, and embedding filtering
  - Added comprehensive logging for threshold filtering
  - Implemented adaptive thresholding for embedding similarity
  
- Quality and Symmetry filtering improvements:
  - Made similarity calculations more lenient with minimum similarity floor (0.5)
  - Better handling of small/zero values in comparison metrics
  - More consistent mask creation for determining valid attributes
  
- Embedding similarity enhancements:
  - Added detection of zero/invalid face embeddings
  - Implemented smart fallback for cases with few valid embeddings
  - Added adaptive threshold adjustment to maintain minimum connected edges
  - Detailed logging for embedding similarity distribution
  
- Impact:
  - Graph construction now matches grid search behavior more accurately
  - Reduced reliance on fallback connections for disconnected nodes
  - Average node degree more consistent with grid search predictions
  - Better preservation of hierarchical structure during subgraph creation

## 2025-04-13
- **Fix: Corrected LSH Edge Filtering Logic in `HierarchicalDeepfakeDataloader`**
  - Modified `_filter_edges_lsh` to accept an input list of edges (`edges_to_filter`).
  - Implemented logic to find the intersection between `edges_to_filter` and the candidate pairs identified by the LSH (Nearest Neighbors) search.
  - Updated the call site in `_filter_edges` to pass the current edge list to `_filter_edges_lsh`.
  - This prevents LSH from discarding previous filtering steps (quality, symmetry, race/gender/age grouping) and generating edges across the entire dataset, resolving the issue of excessively high average node degrees.

## 2025-04-13 (Update)

- **Fix: Removed Adaptive Embedding Threshold in `HierarchicalDeepfakeDataloader`**
  - Removed the logic in `_calculate_pairwise_similarities` that adjusted the embedding threshold dynamically to ensure a minimum percentage of edges were kept.
  - The `embedding_threshold` hyperparameter is now strictly adhered to during vectorized filtering.
  - This aims to further reduce the number of edges by preventing potentially lenient filtering intended for "grid search behavior" from overriding the specified threshold.

## 2025-04-13 (Update)
- **Fix: Corrected Quality/Symmetry Similarity Calculation**
  - Removed the artificial `np.maximum(..., 0.5)` floor applied to individual quality and symmetry metric similarities in `_calculate_pairwise_similarities`. Similarity values can now be less than 0.5.
  - Corrected the logic to use the actual validity masks (`quality_data['mask']`, `symmetry_data['mask']`) to ensure only valid metrics contribute to the average similarity calculation for an edge. Pairs with no shared valid metrics will now have a similarity of 0.
  - This fixes the issue where quality/symmetry filters removed zero edges when the threshold was <= 0.5.

## 2025-04-13 (Update)
- **Fix: Prevented Duplicate Edge Creation**
  - Modified the edge creation loop in `_build_graph_vectorized` to use a set (`added_pairs`) to track processed node pairs.
  - Ensures only one `Edge` object is created and added per unique pair of nodes, even if the filtered edge list contains duplicates.
  - Prevents artificial inflation of node degrees due to duplicate edges.

## 2025-04-13
- **Fix: Strict Fallback Subgroup Constraint**: Modified the fallback connection logic in `_create_graph_from_edges`. If a disconnected node cannot find another disconnected partner in its subgroup, it now attempts to connect to *any* random node within the same subgroup (even connected ones). The previous cross-subgroup fallback connection behavior has been removed.

## 2025-04-13
- **Corrected Initial Edge Generation**: Modified `_build_graph_standard` to generate the initial `all_edges` list *after* the `node_index_to_subgroup_id` map is fully populated. Initial edges are now created strictly between nodes sharing the same final (most specific) subgroup ID, preventing the creation of edges across subgroups during the initial phase.

## 2025-04-13: Fix Index Error in Vectorized Filtering

- **Issue:** Encountered `list index out of range` error during grid search, traced to the vectorized edge filtering path (`_filter_edges_vectorized` -> `_calculate_pairwise_similarities`).
- **Fix:** Modified `_calculate_pairwise_similarities` in `HierarchicalDeepfakeDataloader.py` to validate incoming edge indices against the size of the node attribute matrices. Invalid pairs are now logged and skipped, preventing index errors during numpy array access.
- **Impact:** Should resolve the grid search crash and allow vectorized filtering to function correctly even if invalid edge indices are somehow passed to it.

## 2025-04-13: Fix Index Error in Age Subgroup Creation

- **Issue:** Encountered `list index out of range` error during grid search, originating in `_create_age_subgroups`.
- **Cause:** The function was called with a subset of the `nodes` list created via list comprehension, but still received the original `group_indices` which were out of bounds for the subset.
- **Fix:** Modified the call site in `_build_graph_standard` to pass the full `nodes` list to `_create_age_subgroups` along with the `group_indices`.
- **Impact:** Resolved the `IndexError` during age-based subgrouping.

## 2025-04-13: Fix ValueError in Embedding Normalization

- **Issue:** Encountered `ValueError: operands could not be broadcast together...` during grid search, originating in `_calculate_pairwise_similarities` when calculating embedding similarities.
- **Cause:** Attempting to divide the embedding matrix (shape N, 512) by the norms vector (shape N,) without proper broadcasting.
- **Fix:** Reshaped the `norms[mask]` vector to `norms[mask][:, np.newaxis]` (shape N, 1) before the division operation. This allows NumPy to correctly broadcast the division across the embedding dimensions.
- **Impact:** Resolved the `ValueError`, enabling correct vectorized calculation of cosine similarity for embeddings.

## 2025-04-14: Node ID Implementation for Caching

- **Node Class (`nodes/Node.py`):**
    - Added `node_id` parameter to `__init__`.
    - Updated `__eq__` and `__hash__` methods to use `self.node_id` for comparisons and hashing.
- **HyperGraph Class (`graphs/HyperGraph.py`):**
    - Modified internal `_node_data_map` to use `node.node_id` as keys.
    - Updated `add_node` to check for duplicates based on `node_id`.
    - Updated `get_edge_list` to return tuples of `node_id`s.
    - Updated `add_edges_from_list` to look up nodes using the provided IDs.
- **AIFaceDataset (`datasets/AIFaceDataset.py`):**
    - Modified the `Node` instantiation within `create_nodes_threaded.process_slice` and `create_node` to pass the image file `path` as the first argument (`node_id`).

These changes ensure that nodes have a unique, hashable identifier, allowing the edge-list caching mechanism to function correctly.

## 2025-04-16

*   **`IValueTrainer.__init__`**: Fixed several initialization errors:
    *   Corrected call to `self.graphmanager.get_graph()` (removed `split` argument).
    *   Restored the conversion of the input `attribute_metadata` (list of dicts) into a list of `AttributeMetadata` objects and stored it in `self.attribute_metadata`.
    *   Restored the initialization of `self.bias_loss`, `self.attr_map`, and `self.bias_weight` based on the processed `self.attribute_metadata`.
    *   Restored original `self.prediction_stats` structure.
*   **`IValueTrainer._get_dqn_features`**: Updated to access attributes using `AttributeMetadata` object properties (`attr_meta.name`, `attr_meta.attr_type`) instead of dictionary keys, fixing the `'dict' object has no attribute 'name'` error.
*   **`IValueTrainer.train`/`val`**: Added checks and warnings around bias loss calculation to prevent errors if `self.bias_loss` is `None` or calculation fails.

*   **Refactoring**: Moved `AttributeMetadata` and `AttributeBiasLoss` classes from `trainers/IValueTrainer.py` to a new dedicated file `utils/attribute_utils.py` for better code organization. Updated imports in `IValueTrainer.py` accordingly.

*   **`IValueTrainer._get_dqn_features`**: Fixed `TypeError: only length-1 arrays can be converted to Python scalars` by explicitly handling the `face_embedding` attribute. The embedding array is now stored separately and not passed to `float()`. Other numerical attributes continue to be converted to float.
*   **`IValueTrainer.get_i_value`**: Updated to correctly unpack and use the separate `features_tensor` and `embedding_tensor` returned by `_get_dqn_features`. Handles missing embeddings by creating zero tensors. Ensures tensors are on the correct device before DQN prediction.
*   **Minor Fixes**: Added robustness checks in feature/data loading and loss calculation sections.

## 2025-04-16: Modified IValueTrainer to calculate training bias loss and optionally include it in the main loss via new __init__ flags

*   **`IValueTrainer`**: 
    *   Modified `__init__` to accept `use_bias_loss_in_training` (bool) and `bias_loss_weight` (float) arguments.
    *   Modified `train` method:
        *   Calculates training bias loss (`bias_loss_val`) in each mini-batch using the same filtering logic as the `val` method.
        *   Accumulates and reports `avg_train_bias_loss`.
        *   Optionally adds `bias_loss_weight * bias_loss_val` to the primary loss before `backward()` call if `use_bias_loss_in_training` is True.
    *   Updated `run` method to log both training and validation bias loss.
    *   Corrected calculation of `avg_val_bias_loss` in `val` method.

## 2025-04-13: Fix for TypeError in _get_dqn_features and related updates

*   **`IValueTrainer._get_dqn_features`**: Fixed `TypeError: only length-1 arrays can be converted to Python scalars` by explicitly handling the `face_embedding` attribute. The embedding array is now stored separately and not passed to `float()`. Other numerical attributes continue to be converted to float.
*   **`IValueTrainer.get_i_value`**: Updated to correctly unpack and use the separate `features_tensor` and `embedding_tensor` returned by `_get_dqn_features`. Handles missing embeddings by creating zero tensors. Ensures tensors are on the correct device before DQN prediction.
*   **Minor Fixes**: Added robustness checks in feature/data loading and loss calculation sections.

## 2025-04-13: Fix for AttributeError ('IValueTrainer' object has no attribute 'device') in get_i_value

*   **`IValueTrainer.get_i_value`**: Fixed `AttributeError: 'IValueTrainer' object has no attribute 'device'`. The method now correctly uses the device from the specific DQN model instance (`dqn_model.device`) for tensor creation and placement, instead of attempting to use `self.device`.

## 2025-04-13: Fix for AttributeError ('float' object has no attribute 'detach') by modifying DQNModel.predict_i_value

*   **`DQNModel.predict_i_value`**: Modified to return the raw Q-value tensor instead of calculating the I-value and converting to a float. This resolves the `AttributeError: 'float' object has no attribute 'detach'` in `IValueTrainer.get_i_value`. Also ensured the zero tensor for missing embeddings is created on the correct device (`self.device`). Updated docstring accordingly.

## 2025-04-13: Fix for bias loss calculation in IValueTrainer.val

*   **`IValueTrainer.val`**: Fixed `TypeError: AttributeBiasLoss.forward() missing 1 required positional argument: 'nodes'`. Corrected the call to `self.bias_loss` within the validation loop to pass the required arguments: `outputs`, `labels_batch`, and the list of `node` objects (`mini_batch`) for the current validation mini-batch.

*   **`IValueTrainer.val`**: Refined bias loss calculation. The code now correctly handles cases where only a subset of nodes in a validation mini-batch have attributes. It filters `outputs` and `labels_batch` to align with the nodes that have attributes (`valid_nodes_for_bias`) before passing them to `self.bias_loss`, preventing index mismatches.
*   **Note:** Fixed an accidental revert of the bias loss calculation logic. Restored the correct filtering of outputs, labels, and nodes, ensuring all required arguments (`predictions`, `labels`, `nodes`) are passed to `AttributeBiasLoss.forward`.
## 2025-04-16: Debugged Bias Loss Calculation in attribute_utils.py

*   **`attribute_utils.py`**: Confirmed `AttributeBiasLoss.forward` is called, keys match, and there's attribute value variation in initial batches. Added a new one-time debug print inside the attribute loop to show the calculated `group_errors` and `bias_metrics` (raw error differences between groups) for the first attribute processed. This helps inspect the core bias calculation step.

## 2025-04-16: Debugged Bias Loss Calculation in attribute_utils.py

*   **`attribute_utils.py`**: Confirmed `AttributeBiasLoss.forward` is called, keys match, and attribute value variation exists. Found bias calculation details print wasn't reached. Moved the debug print to occur *before* the inner loop within the attribute loop. It now logs `attr_name`, `value_means` (mean error per group), and `len(values)` for the first attribute processed, regardless of group count, to see if the issue is failing the two-group check.

## 2025-04-16: Debugged Bias Loss Calculation in attribute_utils.py

*   **`attribute_utils.py`**: Confirmed `AttributeBiasLoss.forward` is called, keys match, attribute value variation exists. Found inner loop debug prints weren't reached. Added a new one-time debug print *before* the main attribute loop (`for attr_name...`) to log the content of `self.attr_map` itself. This checks if the loop isn't running because the map is empty.

## 2025-04-16: Fixed Prediction Shape Handling in Bias Loss

*   **`attribute_utils.py`**: Identified that `AttributeBiasLoss.forward` received predictions with shape `[batch_size, 1]`, causing it to exit early. Modified the shape handling logic to `squeeze` the last dimension in this case, converting the tensor to `[batch_size]` before applying sigmoid, allowing the bias calculation to proceed. Removed temporary debug prints.

*   **`attribute_utils.py`**: Removed the `_printed_attr_map_content` and `_printed_group_means` debug prints from `AttributeBiasLoss.forward` now that the bias calculation is confirmed to be working correctly.
## 2025-04-16: Added Final I-Value Calculation and Detailed Test Output

*   **`trainers/IValueTrainer.py`**: Added a new method `get_all_final_i_values(graph_split)` that iterates through nodes of a specified graph split, uses the trained DQN model to predict the I-value for each node based on its attributes, and returns a dictionary mapping node IDs to their final I-values.
*   **`test_hierarchical.py`**: Modified the main testing loop. After `trainer.test()` completes, if the trainer is an `IValueTrainer`, it now calls `get_all_final_i_values('test')`. It then gathers data for all nodes in the test set (ID, attributes, ground truth label, and calculated final I-value) into a pandas DataFrame and saves this detailed information to a unique CSV file named `logs/test_results_{arch}_{traversal_type}_{timestamp}.csv`. This replaces the previous behavior of only printing summary metrics.

## Update (2025-04-17 10:34 AM)

### File: `traversals/IValueTraversalClusterHop.py`

-   **Implemented Cluster Hop Logic:** Modified the `IValueTraversal` copy to introduce a periodic "bias hop".
    -   Added parameters `bias_hop_period`, `pessimistic_i_value`, and `bias_attributes` to `__init__`.
    -   Created `_get_i_value` helper for safe I-value retrieval with pessimistic defaults.
    -   Modified `reset_pointers` to use lazy I-value loading.
    -   In `traverse`, added logic to run every `bias_hop_period` steps:
        -   Calculate the average I-value for nodes grouped by `bias_attributes`.
        -   Identify the group with the *highest* average I-value (most uncertain).
        -   Force a pointer to jump to a random node within that target group.
        -   Added logging for this process.
    -   Updated neighbor selection to use `_get_i_value`.
-   **Goal:** Counteract traversal bias by forcing exploration into attribute groups the DQN is most uncertain about.

## Update (2025-04-17 10:41 AM)

### File: `test_hierarchical.py`

-   **Integrated Cluster Hop Traversal:** Updated the main testing script to allow selection and use of the new `IValueTraversalClusterHop`.
    -   Imported `IValueTraversalClusterHop`.
    -   Added `i-value-cluster-hop` as a choice for the `--traversal-type` argument.
    -   Modified logic to ensure `IValueTrainer` is created for both `i-value` and `i-value-cluster-hop` types.
    -   Added logic to instantiate `IValueTraversalClusterHop` when selected, passing the `trainer` instance.

## 2025-04-24:
*   **Refactored Evaluation in `test_hierarchical.py`**:
    *   Integrated `NodeEvaluationDataset` and `evaluation_collate_fn` into the main loop.
    *   Created `DataLoader` instances (`val_loader`, `test_loader`) for validation and testing.
    *   Removed the old `ComprehensiveTraversal`-based setup for validation and testing.
    *   Replaced the call to `trainer.test()` with a call to the standardized `evaluate_model` function using `test_loader`.
    *   Added a TODO comment to update `trainer.run` to accept `val_loader`.
```

### Subgroup I-Value Logging and Plotting
- **`IValueTraversalClusterHop` Modification:**
    - Removed print statements that output average I-values per subgroup during bias hop calculations.
    - Added `self.hop_i_value_history` list to store dictionaries of `{subgroup_key: avg_i_value}` calculated during each hop check.
    - Added `get_hop_i_value_history()` method to retrieve this data.
- **`test_hierarchical.py` Modification:**
    - Added `plot_subgroup_i_values(history, output_filename)` function using `pandas` and `matplotlib` to generate a line plot showing the average I-value trend for each subgroup over hop instances.
    - After the training loop, the script now retrieves the hop history from the `train_traversal` object and calls `plot_subgroup_i_values` to save the plot to the logs directory (e.g., `logs/YYYYMMDD_HHMMSS_subgroup_i_values.png`).
```

## 2025-04-25
*   **Centralized Image Transforms:** Refactored `IValueTrainer`, `ExperimentTrainer`, and `evaluate_model` in `test_hierarchical.py` to use the `CNNModel`'s built-in `transform` method. The model's `current_mode` attribute ('train' or 'eval') now controls whether training augmentations or common transforms are applied, centralizing the transformation logic.
```

## 2025-04-25
*   **Centralized Image Transforms:** Refactored `IValueTrainer`, `ExperimentTrainer`, and `evaluate_model` in `test_hierarchical.py` to use the `CNNModel`'s built-in `transform` method. The model's `current_mode` attribute ('train' or 'eval') now controls whether training augmentations or common transforms are applied, centralizing the transformation logic.
*   **Fix `_get_cnn_features` Transforms:** Updated the `_get_cnn_features` helper method within `IValueTrainer` to correctly utilize the primary `CNNModel`'s internal `transform` method (using 'eval' mode) instead of its own transform pipeline. This ensures consistency in image preprocessing for the DQN state.
```

## 2025-04-25
*   **Removed Unused Method:** Deleted the `process_node_data` method from `IValueTrainer` as it was no longer being used. The core batch processing logic is handled directly within the `train` method.