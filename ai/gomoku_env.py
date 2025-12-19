"""
Gomoku Environment - LightZero Style
오목 게임 환경 구현 (15x15 보드)
"""
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class GameState:
    """게임 상태를 담는 데이터 클래스"""
    board: np.ndarray
    current_player: int
    done: bool
    winner: int  # 0: 무승부, 1: 흑, 2: 백


class GomokuEnv:
    """
    오목 게임 환경
    LightZero의 GomokuEnv를 참고하여 구현
    """
    
    def __init__(self, board_size: int = 15):
        """
        Args:
            board_size: 보드 크기 (default: 15x15)
        """
        self.board_size = board_size
        self.players = [1, 2]  # 1: 흑, 2: 백
        self.action_space_size = board_size * board_size
        self.reset()
    
    def reset(self, start_player_index: int = 0, init_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        게임 초기화
        
        Args:
            start_player_index: 시작 플레이어 인덱스 (0 또는 1)
            init_state: 초기 보드 상태 (선택적)
        
        Returns:
            현재 상태 관측값
        """
        if init_state is not None:
            self.board = init_state.copy()
        else:
            self.board = np.zeros((self.board_size, self.board_size), dtype=np.int32)
        
        self._current_player = self.players[start_player_index]
        self._done = False
        self._winner = 0
        
        return self._get_observation()
    
    @property
    def current_player(self) -> int:
        """현재 플레이어 반환"""
        return self._current_player
    
    @property
    def legal_actions(self) -> List[int]:
        """유효한 액션(빈 칸) 목록 반환"""
        actions = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 0:
                    actions.append(i * self.board_size + j)
        return actions
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        액션 수행
        
        Args:
            action: 착수 위치 (0 ~ board_size*board_size - 1)
        
        Returns:
            observation: 다음 상태
            reward: 보상
            done: 종료 여부
            info: 추가 정보
        """
        if self._done:
            return self._get_observation(), 0, True, {'winner': self._winner}
        
        row = action // self.board_size
        col = action % self.board_size
        
        # 유효성 검사
        if self.board[row, col] != 0:
            raise ValueError(f"Invalid action: position ({row}, {col}) is already occupied")
        
        # 착수
        self.board[row, col] = self._current_player
        
        # 승리 체크
        done, winner = self.get_done_winner()
        self._done = done
        self._winner = winner
        
        # 보상 계산
        reward = 0
        if done:
            if winner == self._current_player:
                reward = 1
            elif winner != 0:
                reward = -1
        
        # 플레이어 전환
        if not done:
            self._current_player = 3 - self._current_player  # 1 <-> 2
        
        return self._get_observation(), reward, done, {'winner': winner}
    
    def get_done_winner(self) -> Tuple[bool, int]:
        """
        게임 종료 및 승자 확인
        
        Returns:
            done: 게임 종료 여부
            winner: 승자 (0: 무승부/진행중, 1: 흑, 2: 백)
        """
        # 5개 연속 체크
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # 가로, 세로, 대각선
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 0:
                    continue
                
                player = self.board[i, j]
                
                for di, dj in directions:
                    count = 1
                    ni, nj = i + di, j + dj
                    
                    while (0 <= ni < self.board_size and 
                           0 <= nj < self.board_size and 
                           self.board[ni, nj] == player):
                        count += 1
                        ni += di
                        nj += dj
                    
                    if count >= 5:
                        return True, player
        
        # 무승부 체크 (보드가 가득 찼는지)
        if len(self.legal_actions) == 0:
            return True, 0
        
        return False, 0
    
    def _get_observation(self) -> np.ndarray:
        """
        관측값 생성 (LightZero 스타일: 3채널)
        - 채널 0: 현재 플레이어의 돌
        - 채널 1: 상대 플레이어의 돌
        - 채널 2: 현재 플레이어 표시
        
        Returns:
            observation: (3, board_size, board_size) 형태의 배열
        """
        obs = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        
        # 현재 플레이어의 돌
        obs[0] = (self.board == self._current_player).astype(np.float32)
        # 상대 플레이어의 돌
        obs[1] = (self.board == (3 - self._current_player)).astype(np.float32)
        # 현재 플레이어 표시 (흑=1, 백=0)
        obs[2] = np.full((self.board_size, self.board_size), 
                         1.0 if self._current_player == 1 else 0.0, 
                         dtype=np.float32)
        
        return obs
    
    def get_state_for_mcts(self) -> dict:
        """MCTS 시뮬레이션을 위한 상태 반환"""
        return {
            'init_state': self.board.copy(),
            'start_player_index': 0 if self._current_player == 1 else 1
        }
    
    def render(self) -> str:
        """보드를 문자열로 렌더링"""
        symbols = {0: '.', 1: 'X', 2: 'O'}
        lines = []
        
        # 열 번호
        header = '   ' + ' '.join([f'{i:2d}' for i in range(self.board_size)])
        lines.append(header)
        
        for i in range(self.board_size):
            row = f'{i:2d} '
            row += ' '.join([f' {symbols[self.board[i, j]]}' for j in range(self.board_size)])
            lines.append(row)
        
        return '\n'.join(lines)
    
    def copy(self) -> 'GomokuEnv':
        """환경 복사"""
        new_env = GomokuEnv(self.board_size)
        new_env.board = self.board.copy()
        new_env._current_player = self._current_player
        new_env._done = self._done
        new_env._winner = self._winner
        return new_env
    
    def to_dict(self) -> dict:
        """JSON 직렬화를 위한 딕셔너리 변환"""
        # numpy 타입을 Python 기본 타입으로 변환
        board_list = [[int(cell) for cell in row] for row in self.board]
        return {
            'board': board_list,
            'current_player': int(self._current_player),
            'done': bool(self._done),
            'winner': int(self._winner),
            'legal_actions': [int(a) for a in self.legal_actions]
        }


if __name__ == '__main__':
    # 테스트
    env = GomokuEnv(9)
    env.reset()
    print("GomokuEnv created successfully!")
    env.step(40)
    print(env.render())
