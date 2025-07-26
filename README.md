# player_ReId
Player Re-Identification in Sports Footage

This project detects and tracks players in a sports video. Each player is assigned a unique ID that stays the same even when they leave and re-enter the frame.

Requirements:
- Python 3
- Install dependencies with:
pip install -r requirements.txt

How to Run:
1. Place the input video (15sec_input_720p.mp4) in the project folder.
2. Make sure models/player_detector.pt exists.
3. Run:
python main.py
4. Output will be saved to outputs/output_with_ids.mp4.

Files:
- main.py: Main code for detection and tracking
- models/player_detector.pt: YOLO model file
- outputs/output_with_ids.mp4: Output video
- requirements.txt: Dependency list
- README.md: This file

Notes:
IDs are displayed at the bottom of each player box. They are assigned in the order players first appear. The tracker is tuned to try to keep IDs consistent when players leave and return.
