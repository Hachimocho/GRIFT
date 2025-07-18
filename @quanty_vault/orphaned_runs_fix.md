# Orphaned Queued Runs Fix - Quanty 8

## Problem
When something goes wrong while runs are queued, they can remain listed as "queued" in their status but will never actually be run. This creates confusion for users who see runs stuck in queued status indefinitely.

## Solution
Implemented a comprehensive system to detect and handle orphaned queued runs:

### Backend Changes (GPU Queue Manager)

1. **Added `check_orphaned_queued_runs()` method**:
   - Compares runs marked as "queued" with the actual queue
   - Identifies runs that are marked as queued but not in the actual queue
   - Marks orphaned runs as "failed" with appropriate error message
   - Returns list of orphaned run IDs for logging
   - **FIXED**: Now checks both in-memory metadata and saved files
   - **ENHANCED**: Added comprehensive debug logging

2. **Enhanced GPU monitoring loop**:
   - Added periodic check for orphaned runs every 30 seconds
   - Uses a counter to avoid checking too frequently
   - Integrated seamlessly with existing GPU monitoring

3. **Added API endpoint**:
   - `/api/gpu/check-orphaned` (POST) for manual triggering
   - Returns success status and list of orphaned runs found
   - Includes proper error handling and logging

4. **Enhanced `reconcile_existing_runs()` method**:
   - Now handles both "running" and "queued" runs on startup
   - Marks orphaned queued runs as failed during server restart

### Frontend Changes (Runs Page)

1. **Added JavaScript function `checkOrphanedRuns()`**:
   - Calls the API endpoint to check for orphaned runs
   - Automatically refreshes page if orphaned runs are found
   - Includes proper error handling

2. **Enhanced page initialization**:
   - Runs initial orphaned check when page loads
   - Sets up periodic checking every 60 seconds
   - Integrated with existing auto-refresh logic

### Key Features

1. **Automatic Detection**: Background thread checks every 30 seconds
2. **Manual Triggering**: API endpoint allows manual checks
3. **Frontend Integration**: Page automatically checks when user is viewing runs
4. **Proper Status Updates**: Orphaned runs are marked as "failed" with clear error message
5. **Logging**: Comprehensive logging for debugging and monitoring
6. **File System Check**: Checks both in-memory metadata and saved files
7. **Server Restart Handling**: Handles orphaned runs during server startup

### Error Handling

- Graceful handling of exceptions in all methods
- Proper logging of errors and warnings
- Frontend error handling with console logging
- No impact on normal queue operation if orphaned check fails

### Debug Logging

The enhanced version includes detailed logging:
- Number of runs marked as queued vs actual queue length
- List of actual queue run IDs
- List of orphaned run IDs found
- Individual logging for each orphaned run being marked as failed

### Testing

The fix should:
1. Detect runs that are marked as "queued" but not in the actual queue
2. Mark them as "failed" with appropriate error message
3. Update the UI automatically when orphaned runs are found
4. Provide manual API endpoint for testing and debugging
5. Handle both in-memory and file-based orphaned runs

## Files Modified

1. `web_ui/gpu_queue_manager.py` - Added orphaned run detection logic with enhanced debugging
2. `web_ui/app.py` - Added API endpoint for manual orphaned run checks
3. `web_ui/templates/runs.html` - Added frontend JavaScript for periodic checking

## Usage

- **Automatic**: Runs every 30 seconds in background, 60 seconds on frontend
- **Manual**: Call `/api/gpu/check-orphaned` POST endpoint
- **Frontend**: Automatically checks when user is on runs page
- **Debug**: Check logs for detailed information about orphaned run detection

## Debugging

To test the orphaned run detection:
1. Start the web UI
2. Check the logs for debug messages from `check_orphaned_queued_runs()`
3. Look for messages like "Checking for orphaned runs: X runs marked as queued but not in actual queue"
4. If orphaned runs are found, they will be marked as failed and logged 