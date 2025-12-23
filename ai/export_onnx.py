"""
PyTorch 모델을 ONNX로 변환하는 스크립트
"""
import torch
import torch.nn as nn
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.network import PolicyValueNetwork


def export_to_onnx(board_size: int = 9, output_path: str = "docs/model.onnx", model_path: str = None):
    """
    PolicyValueNetwork를 ONNX 형식으로 내보내기
    
    Args:
        board_size: 보드 크기
        output_path: ONNX 파일 저장 경로
        model_path: 학습된 모델 경로 (없으면 랜덤 초기화)
    """
    # 모델 생성
    model = PolicyValueNetwork(
        board_size=board_size,
        num_channels=64,
        num_res_blocks=4
    )
    
    # 학습된 모델 로드
    if model_path and os.path.exists(model_path):
        print(f"학습된 모델 로드: {model_path}")
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        print("랜덤 초기화된 모델 사용")
    
    model.eval()
    
    # 더미 입력 생성
    dummy_input = torch.randn(1, 3, board_size, board_size)
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # ONNX로 내보내기 (웹 호환성을 위해 Opset 10 사용 및 차원 고정)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=10,  # 웹 호환성을 위해 10으로 낮춤
        do_constant_folding=True,
        input_names=['input'],
        output_names=['policy', 'value']
    )
    
    print(f"ONNX 모델 저장 완료: {output_path}")
    print(f"보드 크기: {board_size}x{board_size}")
    print(f"입력 형태: (batch, 3, {board_size}, {board_size})")
    print(f"출력: policy ({board_size*board_size}), value (1)")
    
    # 파일 크기 확인
    file_size = os.path.getsize(output_path)
    print(f"파일 크기: {file_size / 1024:.1f} KB")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Export model to ONNX')
    parser.add_argument('--board-size', type=int, default=9, help='Board size')
    parser.add_argument('--output', type=str, default='docs/model.onnx', help='Output path')
    parser.add_argument('--model', type=str, default=None, help='Trained model path')
    
    args = parser.parse_args()
    
    export_to_onnx(args.board_size, args.output, args.model)

