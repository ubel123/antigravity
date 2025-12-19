"""
AlphaZero Training Pipeline
자가 대국 및 학습 파이프라인
"""
import os
import numpy as np
import torch
from typing import List, Tuple, Dict
from collections import deque
import random

from .gomoku_env import GomokuEnv
from .mcts import MCTS
from .network import PolicyValueFunction


class SelfPlayData:
    """자가 대국 데이터"""
    
    def __init__(self, state: np.ndarray, mcts_probs: np.ndarray, winner: int):
        self.state = state
        self.mcts_probs = mcts_probs
        self.winner = winner


class AlphaZeroTrainer:
    """
    AlphaZero 훈련 클래스
    LightZero의 train_alphazero 참고
    """
    
    def __init__(
        self,
        board_size: int = 15,
        num_simulations: int = 100,
        num_episodes: int = 100,
        batch_size: int = 256,
        buffer_size: int = 10000,
        learning_rate: float = 0.001,
        temperature: float = 1.0,
        temp_threshold: int = 30,
        model_save_path: str = 'ai/models',
        device: str = None
    ):
        """
        Args:
            board_size: 보드 크기
            num_simulations: MCTS 시뮬레이션 횟수
            num_episodes: 자가 대국 에피소드 수
            batch_size: 학습 배치 크기
            buffer_size: 리플레이 버퍼 크기
            learning_rate: 학습률
            temperature: 탐색 온도
            temp_threshold: 온도를 낮추는 임계 수
            model_save_path: 모델 저장 경로
            device: 연산 디바이스
        """
        self.board_size = board_size
        self.num_simulations = num_simulations
        self.num_episodes = num_episodes
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.temp_threshold = temp_threshold
        self.model_save_path = model_save_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 환경 초기화
        self.env = GomokuEnv(board_size)
        
        # 정책-가치 네트워크
        self.policy_value_fn = PolicyValueFunction(board_size, device=self.device)
        
        # MCTS
        self.mcts = MCTS(num_simulations=num_simulations)
        
        # 리플레이 버퍼
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # 옵티마이저
        self.optimizer = torch.optim.Adam(
            self.policy_value_fn.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )
        
        # 디렉토리 생성
        os.makedirs(model_save_path, exist_ok=True)
        
        # 학습 통계
        self.stats = {
            'episode': 0,
            'total_loss': [],
            'policy_loss': [],
            'value_loss': [],
            'game_lengths': [],
            'win_rate': []
        }
    
    def self_play_game(self) -> List[SelfPlayData]:
        """
        자가 대국 수행
        
        Returns:
            게임 데이터 리스트
        """
        self.env.reset()
        game_data = []
        move_count = 0
        
        while True:
            # 온도 조정 (게임 초반에는 탐색, 후반에는 최적 행동)
            temp = self.temperature if move_count < self.temp_threshold else 0.1
            
            # MCTS로 행동 선택
            action, action_probs = self.mcts.get_action_probs(
                self.env,
                self.policy_value_fn,
                temperature=temp,
                add_noise=True
            )
            
            # 데이터 저장
            state = self.env._get_observation()
            game_data.append({
                'state': state,
                'mcts_probs': action_probs,
                'current_player': self.env.current_player
            })
            
            # 행동 수행
            _, _, done, info = self.env.step(action)
            move_count += 1
            
            if done:
                winner = info['winner']
                break
        
        # 승자에 따라 가치 레이블 할당
        play_data = []
        for data in game_data:
            if winner == 0:
                value = 0.0  # 무승부
            elif winner == data['current_player']:
                value = 1.0  # 승리
            else:
                value = -1.0  # 패배
            
            play_data.append(SelfPlayData(
                state=data['state'],
                mcts_probs=data['mcts_probs'],
                winner=value
            ))
        
        return play_data, move_count
    
    def train_step(self) -> Tuple[float, float, float]:
        """
        학습 스텝 수행
        
        Returns:
            total_loss, policy_loss, value_loss
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0, 0, 0
        
        # 미니배치 샘플링
        batch = random.sample(list(self.replay_buffer), self.batch_size)
        
        states = np.array([d.state for d in batch])
        target_policies = np.array([d.mcts_probs for d in batch])
        target_values = np.array([d.winner for d in batch])
        
        # 학습
        total_loss, policy_loss, value_loss = self.policy_value_fn.train_step(
            states, target_policies, target_values, self.optimizer
        )
        
        return total_loss, policy_loss, value_loss
    
    def train(self, num_iterations: int = 10, games_per_iteration: int = 10, train_steps_per_iteration: int = 50):
        """
        전체 학습 루프
        
        Args:
            num_iterations: 반복 횟수
            games_per_iteration: 반복당 자가 대국 횟수
            train_steps_per_iteration: 반복당 학습 스텝 수
        """
        for iteration in range(num_iterations):
            print(f"\n=== Iteration {iteration + 1}/{num_iterations} ===")
            
            # 자가 대국
            total_moves = 0
            for game in range(games_per_iteration):
                play_data, move_count = self.self_play_game()
                self.replay_buffer.extend(play_data)
                total_moves += move_count
                print(f"  Game {game + 1}: {move_count} moves")
            
            self.stats['game_lengths'].append(total_moves / games_per_iteration)
            
            # 학습
            total_loss_sum = 0
            policy_loss_sum = 0
            value_loss_sum = 0
            train_count = 0
            
            for _ in range(train_steps_per_iteration):
                total_loss, policy_loss, value_loss = self.train_step()
                if total_loss > 0:
                    total_loss_sum += total_loss
                    policy_loss_sum += policy_loss
                    value_loss_sum += value_loss
                    train_count += 1
            
            if train_count > 0:
                avg_total_loss = total_loss_sum / train_count
                avg_policy_loss = policy_loss_sum / train_count
                avg_value_loss = value_loss_sum / train_count
                
                self.stats['total_loss'].append(avg_total_loss)
                self.stats['policy_loss'].append(avg_policy_loss)
                self.stats['value_loss'].append(avg_value_loss)
                
                print(f"  Loss: Total={avg_total_loss:.4f}, Policy={avg_policy_loss:.4f}, Value={avg_value_loss:.4f}")
            
            # 모델 저장
            if (iteration + 1) % 5 == 0:
                self.save_model(f"model_iter_{iteration + 1}.pt")
        
        # 최종 모델 저장
        self.save_model("model_final.pt")
        print("\nTraining completed!")
    
    def save_model(self, filename: str):
        """모델 저장"""
        path = os.path.join(self.model_save_path, filename)
        self.policy_value_fn.save(path)
        print(f"  Model saved: {path}")
    
    def load_model(self, filename: str):
        """모델 로드"""
        path = os.path.join(self.model_save_path, filename)
        self.policy_value_fn.load(path)
        print(f"Model loaded: {path}")
    
    def get_stats(self) -> Dict:
        """학습 통계 반환"""
        return self.stats


class AIPlayer:
    """AI 플레이어 (추론용)"""
    
    def __init__(
        self,
        board_size: int = 15,
        num_simulations: int = 100,
        model_path: str = None,
        device: str = None
    ):
        self.board_size = board_size
        self.policy_value_fn = PolicyValueFunction(board_size, model_path, device)
        self.mcts = MCTS(num_simulations=num_simulations)
    
    def get_action(self, env: GomokuEnv, temperature: float = 0.1) -> int:
        """
        최적 행동 선택
        
        Args:
            env: 현재 게임 환경
            temperature: 탐색 온도 (낮을수록 deterministic)
        
        Returns:
            선택된 행동
        """
        action, _ = self.mcts.get_action_probs(
            env,
            self.policy_value_fn,
            temperature=temperature,
            add_noise=False
        )
        return action


def ai_battle(
    player1: AIPlayer,
    player2: AIPlayer,
    board_size: int = 15,
    verbose: bool = True
) -> Tuple[int, List[int]]:
    """
    두 AI 플레이어 대결
    
    Args:
        player1: 흑 플레이어
        player2: 백 플레이어
        board_size: 보드 크기
        verbose: 출력 여부
    
    Returns:
        winner: 승자 (0: 무승부, 1: 흑, 2: 백)
        moves: 착수 기록
    """
    env = GomokuEnv(board_size)
    env.reset()
    
    players = {1: player1, 2: player2}
    moves = []
    
    while True:
        current_player = env.current_player
        player = players[current_player]
        
        action = player.get_action(env)
        moves.append(action)
        
        if verbose:
            row, col = action // board_size, action % board_size
            print(f"Player {current_player}: ({row}, {col})")
        
        _, _, done, info = env.step(action)
        
        if done:
            winner = info['winner']
            if verbose:
                if winner == 0:
                    print("Draw!")
                else:
                    print(f"Player {winner} wins!")
                print(env.render())
            return winner, moves
