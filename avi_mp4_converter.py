# avi video to mp4 video converter

import cv2

def avi_to_mp4(avi_file, mp4_file):
    video = cv2.VideoCapture(avi_file)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(mp4_file, fourcc, fps, (width, height))
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        out.write(frame)
    
    video.release()
    out.release()

def mp4_to_avi(mp4_file, avi_file):
    video = cv2.VideoCapture(mp4_file)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(avi_file, fourcc, fps, (width, height))
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
        out.write(frame)
    
    video.release()
    out.release()

if __name__ == '__main__':
    avi_to_mp4('test2.avi', 'test.mp4')
    print('Video dönüştürme işlemi başarılı')