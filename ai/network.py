"""
Policy-Value Network for AlphaZero
PyTorch 기반 신경망 구현
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class ResidualBlock(nn.Module):
    """잔차 블록 (Residual Block)"""
    
    def __init__(self, num_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class PolicyValueNetwork(nn.Module):
    """
    Policy-Value 네트워크
    AlphaZero의 신경망 구조 구현
    """
    
    def __init__(
        self,
        board_size: int = 15,
        num_channels: int = 64,
        num_res_blocks: int = 4
    ):
        """
        Args:
            board_size: 보드 크기
            num_channels: CNN 채널 수
            num_res_blocks: 잔차 블록 수
        """
        super().__init__()
        self.board_size = board_size
        self.action_size = board_size * board_size
        
        # 입력 레이어 (3채널 -> num_channels)
        self.conv_input = nn.Conv2d(3, num_channels, 3, padding=1)
        self.bn_input = nn.BatchNorm2d(num_channels)
        
        # 잔차 블록
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_channels) for _ in range(num_res_blocks)
        ])
        
        # Policy Head
        self.policy_conv = nn.Conv2d(num_channels, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, self.action_size)
        
        # Value Head
        self.value_conv = nn.Conv2d(num_channels, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(board_size * board_size, 64)
        self.value_fc2 = nn.Linear(64, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        순전파
        
        Args:
            x: 입력 텐서 (batch, 3, board_size, board_size)
        
        Returns:
            policy: 정책 확률 분포 (batch, action_size)
            value: 상태 가치 (batch, 1)
        """
        # 공통 특징 추출
        out = F.relu(self.bn_input(self.conv_input(x)))
        
        for res_block in self.res_blocks:
            out = res_block(out)
        
        # Policy Head
        policy = F.relu(self.policy_bn(self.policy_conv(out)))
        policy = policy.view(policy.size(0), -1)
        policy = self.policy_fc(policy)
        policy = F.softmax(policy, dim=1)
        
        # Value Head
        value = F.relu(self.value_bn(self.value_conv(out)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value


class PolicyValueFunction:
    """
    MCTS에서 사용할 정책-가치 함수 래퍼
    """
    
    def __init__(
        self,
        board_size: int = 15,
        model_path: str = None,
        device: str = None
    ):
        """
        Args:
            board_size: 보드 크기
            model_path: 모델 가중치 경로
            device: 연산 디바이스 ('cpu' 또는 'cuda')
        """
        self.board_size = board_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = PolicyValueNetwork(board_size).to(self.device)
        
        if model_path:
            self.load(model_path)
        else:
            self._init_weights()
    
    def _init_weights(self):
        """가중치 초기화"""
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def __call__(self, obs: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        관측값에서 정책과 가치 예측
        
        Args:
            obs: 관측값 (3, board_size, board_size)
        
        Returns:
            action_probs: 행동 확률 분포 (action_size,)
            value: 상태 가치 스칼라
        """
        self.model.eval()
        
        with torch.no_grad():
            x = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            policy, value = self.model(x)
            
            action_probs = policy.cpu().numpy()[0]
            value_scalar = value.cpu().numpy()[0, 0]
        
        return action_probs, value_scalar
    
    def predict_batch(self, obs_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """배치 예측"""
        self.model.eval()
        
        with torch.no_grad():
            x = torch.FloatTensor(obs_batch).to(self.device)
            policy, value = self.model(x)
            
            return policy.cpu().numpy(), value.cpu().numpy()
    
    def train_step(
        self,
        states: np.ndarray,
        target_policies: np.ndarray,
        target_values: np.ndarray,
        optimizer: torch.optim.Optimizer
    ) -> Tuple[float, float, float]:
        """
        학습 스텝
        
        Args:
            states: 상태 배치 (batch, 3, board_size, board_size)
            target_policies: 목표 정책 (batch, action_size)
            target_values: 목표 가치 (batch, 1)
            optimizer: 옵티마이저
        
        Returns:
            total_loss, policy_loss, value_loss
        """
        self.model.train()
        
        states_t = torch.FloatTensor(states).to(self.device)
        target_policies_t = torch.FloatTensor(target_policies).to(self.device)
        target_values_t = torch.FloatTensor(target_values).to(self.device)
        
        # 순전파
        pred_policies, pred_values = self.model(states_t)
        
        # 손실 계산
        policy_loss = -torch.mean(torch.sum(target_policies_t * torch.log(pred_policies + 1e-8), dim=1))
        value_loss = F.mse_loss(pred_values.squeeze(), target_values_t.squeeze())
        total_loss = policy_loss + value_loss
        
        # 역전파
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        return total_loss.item(), policy_loss.item(), value_loss.item()
    
    def save(self, path: str):
        """모델 저장"""
        torch.save(self.model.state_dict(), path)
    
    def load(self, path: str):
        """모델 로드"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
