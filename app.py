from flask import Flask, request, jsonify
from model import SignLanguageModel
from api_handler import get_transit_route, parse_route_info
import torch
import utils
from dataset import preprocess_frames  # 데이터셋 전처리 함수
from dotenv import load_dotenv
import os

# .env 파일의 환경 변수를 로드합니다.
load_dotenv()

app = Flask(__name__)

# 환경 변수에서 ODsay API 키를 가져옵니다.
API_KEY = os.getenv("ODSAY_API_KEY")

# 수어 인식 모델 초기화 및 로드
model = SignLanguageModel()
model.load_state_dict(torch.load('model.pth'))  # 사전 학습된 모델의 가중치 로드
model.eval()

@app.route('/api/sign_to_transit', methods=['POST'])
def sign_to_transit():
    # 클라이언트로부터 동영상 경로를 전달받습니다.
    video_path = request.json.get('video_path')

    # 동영상 데이터를 프레임으로 전처리
    frames = utils.load_video(video_path)
    preprocessed_frames = preprocess_frames(frames)

    # 수어 인식 모델을 통한 텍스트 예측
    inputs = torch.tensor(preprocessed_frames, dtype=torch.float32).unsqueeze(0)  # 배치 차원 추가
    with torch.no_grad():
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)

    # 예측된 텍스트 (출발지와 도착지) 사용
    # 여기에 매핑 로직 필요: predicted 텍스트를 (위도, 경도)로 변환
    start_coord = (37.5665, 126.9780)  # 예: 서울시청
    end_coord = (37.5080, 127.0620)    # 예: 강남역

    # ODsay API를 통해 대중교통 경로 정보 가져오기
    route_data = get_transit_route(API_KEY, start_coord, end_coord)
    route_info = parse_route_info(route_data)

    return jsonify({'route_info': route_info})

if __name__ == '__main__':
    app.run(debug=True)
