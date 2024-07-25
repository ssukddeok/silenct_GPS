import cv2
import numpy as np

def load_video(video_path):
    """
    동영상 파일에서 프레임을 로드합니다.

    Parameters:
    - video_path (str): 동영상 파일의 경로

    Returns:
    - list: 프레임 이미지 배열
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def preprocess_frames(frames, size=(224, 224)):
    """
    프레임을 전처리합니다. 회색조로 변환하고, 크기를 조정하며, 정규화합니다.

    Parameters:
    - frames (list): 프레임 이미지 배열
    - size (tuple): 출력 이미지 크기

    Returns:
    - np.array: 전처리된 프레임 배열
    """
    processed_frames = []
    for frame in frames:
        frame = cv2.resize(frame, size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame / 255.0  # 정규화
        processed_frames.append(frame)
    return np.array(processed_frames)
