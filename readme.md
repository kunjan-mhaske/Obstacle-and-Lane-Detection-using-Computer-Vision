## Obstacle and Lane Detection using YOLO and Lane-Net 

Steps to execute:

1. Extract the "ObstacleAndLaneDetection" project file from the submission.
2. Download the weights of the pre-trained models from https://drive.google.com/drive/folders/1IX1seops2R8XgnBWfePeDMNAPhz2Yf9f?usp=sharing 
3. Extract and save the "models" folder in the LaneDetectionLaneNet folder.
4. Extract and save the "yolo-coco" folder in the ObstacleDetectionYOLO folder.
5. Execute the Test_Detections_On_Video.py file with the following arguments.
    
   python Test_Detections_On_Video.py --input ./test_video/lane_traffic.avi
                                      --output ./OUT/output_video.avi

6. Check the results in the specified output directory or in the OUT directory.


> Please check Project report pdf for more details.