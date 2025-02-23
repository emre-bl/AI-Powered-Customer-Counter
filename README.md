# AI-Powered Customer Counter

## Project Overview
This repository contains an intelligent customer counting system that leverages **YOLOv8 object detection** and **ByteTrack tracking** to monitor customer flow through specific areas. The system analyzes video feeds to count people crossing a user-defined virtual line within a Region of Interest (ROI), making it ideal for retail analytics, space utilization monitoring, and entrance/exit management.

Key Technical Components:
- **YOLOv8-nano** for real-time person detection
- **ByteTrack algorithm** for robust object tracking
- Adaptive ROI management with virtual entry-line calculation
- Frame-skipping optimization for efficient processing

## Features 💡
- **Smart ROI Configuration**: Interactive selection of monitoring area
- **Dynamic Counting Logic**: Entry/exit detection through trajectory analysis
- **Performance Optimization**: 
  - Configurable frame skipping (3x speed boost default)
  - CUDA acceleration support
- **Flexible Operation Modes**:
  - GUI mode with real-time visualization
  - Headless mode for server/cloud deployments
- **Batch Processing**: Analyze multiple videos in sequence
- **Tracking History**: 15-frame trajectory buffer for accurate path prediction
