# Gemini Usage Guide

This document outlines how to use the Gemini CLI to interact with the HyperGraph project, focusing on running tests through the web UI.

## Running Tests via the Web UI

The primary method for running tests is through the web-based interface.

### 1. Start the Web UI

To start the web UI, run the following command in your terminal:

```bash
python web_ui/app.py
```

This will start the Flask server, and you can access the UI in your web browser at `http://localhost:5000`.

### 2. Using the Web UI

The web UI provides the following functionalities:

*   **Dashboard**: The main page shows an overview of saved test configurations and recent test runs.
*   **Configure**: Create, edit, and save test configurations. You can specify parameters such as the model architecture, number of epochs, and batch size.
*   **Runs**: View the status of all test runs, including active and completed runs.
*   **Results**: Compare the results of different test runs.

### 3. Starting a Test Run

To start a new test run:

1.  Navigate to the **Configure** page.
2.  Create a new configuration or select an existing one.
3.  Click the **Start Test Run** button.

This will queue the test run, and you can monitor its progress on the **Runs** page.

### 4. Viewing Test Results

Once a test run is complete, you can view the results on the **Results** page. This includes metrics such as accuracy and loss, as well as any generated visualizations.
