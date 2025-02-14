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