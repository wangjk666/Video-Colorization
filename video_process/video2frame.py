import cv2
import glob
import os

 
 
def video_to_frames(path):
    videoCapture = cv2.VideoCapture()
    videoCapture.open(path)
    # 帧率
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    # 总帧数
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("fps=", int(fps), "frames=", int(frames))
 
    for i in range(int(frames)):
        ret, frame = videoCapture.read()
        s = "%05d" % i
        image_name = "image_" + s + '.jpg'
        frame_path = os.path.join('/Users/macbook/Desktop/Colorization/video_process/images',image_name)
        cv2.imwrite(frame_path, frame)
    return
 
 
if __name__ == '__main__':
    video_to_frames("/Users/macbook/Desktop/Colorization/video_process/test.mp4")
    

