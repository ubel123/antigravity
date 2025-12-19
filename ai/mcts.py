"""
Monte Carlo Tree Search (MCTS) for AlphaZero
LightZero의 ptree_az.py를 참고하여 구현
"""
import math
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any


class Node:
    """
    MCTS 트리의 노드
    LightZero의 Node 클래스 참고
    """
    
    def __init__(self, parent: Optional['Node'] = None, prior_p: float = 1.0):
        """
        Args:
            parent: 부모 노드
            prior_p: 사전 확률 (policy network 출력)
        """
        self._parent = parent
        self._children: Dict[int, 'Node'] = {}
        self._visit_count = 0
        self._value_sum = 0.0
        self.prior_p = prior_p
    
    @property
    def value(self) -> float:
        """평균 가치 반환"""
        if self._visit_count == 0:
            return 0.0
        return self._value_sum / self._visit_count
    
    @property
    def visit_count(self) -> int:
        """방문 횟수 반환"""
        return self._visit_count
    
    @property
    def children(self) -> Dict[int, 'Node']:
        """자식 노드 딕셔너리 반환"""
        return self._children
    
    @property
    def parent(self) -> Optional['Node']:
        """부모 노드 반환"""
        return self._parent
    
    def is_leaf(self) -> bool:
        """리프 노드인지 확인"""
        return len(self._children) == 0
    
    def is_root(self) -> bool:
        """루트 노드인지 확인"""
        return self._parent is None
    
    def update(self, value: float) -> None:
        """노드 정보 업데이트"""
        self._visit_count += 1
        self._value_sum += value
    
    def update_recursive(self, leaf_value: float, negate: bool = True) -> None:
        """
        재귀적으로 노드 정보 업데이트 (백프로파게이션)
        
        Args:
            leaf_value: 리프 노드의 가치
            negate: 가치 부호 반전 여부 (self-play에서 필요)
        """
        self.update(leaf_value)
        if not self.is_root():
            self._parent.update_recursive(-leaf_value if negate else leaf_value, negate)
    
    def expand(self, action_probs: List[Tuple[int, float]]) -> None:
        """
        노드 확장
        
        Args:
            action_probs: (action, probability) 튜플 리스트
        """
        for action, prob in action_probs:
            if action not in self._children:
                self._children[action] = Node(parent=self, prior_p=prob)


class MCTS:
    """
    Monte Carlo Tree Search
    LightZero의 MCTS 클래스 참고
    """
    
    def __init__(
        self,
        num_simulations: int = 100,
        pb_c_base: float = 19652,
        pb_c_init: float = 1.25,
        root_dirichlet_alpha: float = 0.3,
        root_noise_weight: float = 0.25
    ):
        """
        Args:
            num_simulations: 시뮬레이션 횟수
            pb_c_base: UCB 계산에 사용되는 상수
            pb_c_init: UCB 계산에 사용되는 초기값
            root_dirichlet_alpha: 루트 노드 탐색 노이즈 alpha
            root_noise_weight: 루트 노드 탐색 노이즈 가중치
        """
        self._num_simulations = num_simulations
        self._pb_c_base = pb_c_base
        self._pb_c_init = pb_c_init
        self._root_dirichlet_alpha = root_dirichlet_alpha
        self._root_noise_weight = root_noise_weight
    
    def get_action_probs(
        self,
        env,
        policy_value_fn: Callable,
        temperature: float = 1.0,
        add_noise: bool = True
    ) -> Tuple[int, np.ndarray]:
        """
        MCTS 검색을 수행하고 행동 확률 반환
        
        Args:
            env: 현재 게임 환경
            policy_value_fn: 정책-가치 네트워크 함수 (obs -> (probs, value))
            temperature: 탐색 온도
            add_noise: 루트 노드에 노이즈 추가 여부
        
        Returns:
            action: 선택된 행동
            action_probs: 모든 행동에 대한 확률 분포
        """
        root = Node()
        
        # 루트 노드 확장
        self._expand_leaf_node(root, env, policy_value_fn)
        
        # 탐색 노이즈 추가
        if add_noise:
            self._add_exploration_noise(root)
        
        # MCTS 시뮬레이션
        for _ in range(self._num_simulations):
            sim_env = env.copy()
            self._simulate(root, sim_env, policy_value_fn)
        
        # 행동별 방문 횟수 집계
        action_visits = []
        for action in range(env.action_space_size):
            if action in root.children:
                action_visits.append((action, root.children[action].visit_count))
            else:
                action_visits.append((action, 0))
        
        actions, visits = zip(*action_visits)
        
        # 방문 횟수를 확률로 변환
        visits_array = np.array(visits, dtype=np.float32)
        
        if temperature == 0:
            # Greedy 선택
            action_probs = np.zeros(len(actions))
            action_probs[np.argmax(visits_array)] = 1.0
        else:
            visits_temp = np.power(visits_array, 1.0 / temperature)
            total = np.sum(visits_temp)
            if total > 0:
                action_probs = visits_temp / total
            else:
                action_probs = np.ones(len(actions)) / len(actions)
        
        # 행동 선택
        if temperature == 0:
            action = actions[np.argmax(visits_array)]
        else:
            action = np.random.choice(actions, p=action_probs)
        
        return action, action_probs
    
    def _simulate(
        self,
        node: Node,
        env,
        policy_value_fn: Callable
    ) -> None:
        """
        단일 시뮬레이션 수행 (Selection -> Expansion -> Evaluation -> Backpropagation)
        """
        # Selection: 리프 노드까지 탐색
        while not node.is_leaf():
            action, node = self._select_child(node, env)
            if action is None:
                break
            env.step(action)
        
        # 게임 종료 체크
        done, winner = env.get_done_winner()
        
        if done:
            # 종료 상태의 가치 계산
            if winner == 0:
                leaf_value = 0.0  # 무승부
            elif winner == env.current_player:
                leaf_value = 1.0  # 현재 플레이어 승리
            else:
                leaf_value = -1.0  # 현재 플레이어 패배
        else:
            # Expansion & Evaluation
            leaf_value = self._expand_leaf_node(node, env, policy_value_fn)
        
        # Backpropagation
        node.update_recursive(-leaf_value)
    
    def _expand_leaf_node(
        self,
        node: Node,
        env,
        policy_value_fn: Callable
    ) -> float:
        """
        리프 노드 확장
        
        Returns:
            value: 노드의 가치
        """
        # 정책-가치 네트워크로 확률과 가치 예측
        obs = env._get_observation()
        action_probs, value = policy_value_fn(obs)
        
        # 유효한 행동만 필터링
        legal_actions = env.legal_actions
        if len(legal_actions) == 0:
            return 0.0
        
        # 확률 정규화
        legal_probs = [(a, action_probs[a]) for a in legal_actions]
        total_prob = sum(p for _, p in legal_probs)
        if total_prob > 0:
            legal_probs = [(a, p / total_prob) for a, p in legal_probs]
        else:
            uniform_prob = 1.0 / len(legal_actions)
            legal_probs = [(a, uniform_prob) for a in legal_actions]
        
        # 노드 확장
        node.expand(legal_probs)
        
        return value
    
    def _select_child(self, node: Node, env) -> Tuple[Optional[int], Node]:
        """
        UCB 기반 자식 노드 선택
        """
        legal_actions = env.legal_actions
        
        # 유효한 자식 노드 중 최고 UCB 점수를 가진 노드 선택
        best_score = -float('inf')
        best_action = None
        best_child = node
        
        for action, child in node.children.items():
            if action in legal_actions:
                score = self._ucb_score(node, child)
                if score > best_score:
                    best_score = score
                    best_action = action
                    best_child = child
        
        return best_action, best_child
    
    def _ucb_score(self, parent: Node, child: Node) -> float:
        """
        UCB (Upper Confidence Bound) 점수 계산
        """
        pb_c = math.log((parent.visit_count + self._pb_c_base + 1) / self._pb_c_base) + self._pb_c_init
        prior_score = pb_c * child.prior_p * math.sqrt(parent.visit_count) / (child.visit_count + 1)
        value_score = child.value
        
        return prior_score + value_score
    
    def _add_exploration_noise(self, node: Node) -> None:
        """
        루트 노드에 Dirichlet 노이즈 추가
        """
        actions = list(node.children.keys())
        if len(actions) == 0:
            return
        
        noise = np.random.dirichlet([self._root_dirichlet_alpha] * len(actions))
        
        for i, action in enumerate(actions):
            child = node.children[action]
            child.prior_p = (1 - self._root_noise_weight) * child.prior_p + self._root_noise_weight * noise[i]
