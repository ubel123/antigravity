"""
FastAPI Backend for Gomoku AI Battle
"""
import os
import sys
from pathlib import Path

# 프로젝트 루트 경로 추가
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import uuid
import asyncio
import threading

# AI 모듈 임포트
from ai.gomoku_env import GomokuEnv
from ai.alphazero import AIPlayer, AlphaZeroTrainer
from ai.network import PolicyValueFunction
from ai.mcts import MCTS

app = FastAPI(title="Gomoku AI Battle", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 게임 세션 저장
games: Dict[str, dict] = {}

# 학습 상태
training_status = {
    "is_training": False,
    "current_iteration": 0,
    "total_iterations": 0,
    "current_game": 0,
    "games_per_iteration": 0,
    "losses": []
}


class NewGameRequest(BaseModel):
    board_size: int = 15


class MoveRequest(BaseModel):
    action: int


class AIBattleRequest(BaseModel):
    board_size: int = 15
    num_simulations: int = 50
    delay_ms: int = 500


class TrainRequest(BaseModel):
    board_size: int = 9  # 학습 시 작은 보드 권장
    iterations: int = 5
    games_per_iteration: int = 3
    train_steps: int = 30
    num_simulations: int = 50


@app.get("/")
async def read_root():
    """메인 페이지"""
    return FileResponse(PROJECT_ROOT / "frontend" / "index.html")


@app.post("/api/game/new")
async def new_game(request: NewGameRequest):
    """새 게임 생성"""
    game_id = str(uuid.uuid4())
    env = GomokuEnv(request.board_size)
    env.reset()
    
    games[game_id] = {
        "env": env,
        "moves": [],
        "ai_battle": False
    }
    
    return {
        "game_id": game_id,
        "state": env.to_dict()
    }


@app.get("/api/game/{game_id}")
async def get_game(game_id: str):
    """게임 상태 조회"""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = games[game_id]
    return {
        "game_id": game_id,
        "state": game["env"].to_dict(),
        "moves": game["moves"]
    }


@app.post("/api/game/{game_id}/move")
async def make_move(game_id: str, request: MoveRequest):
    """수동 착수"""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = games[game_id]
    env = game["env"]
    
    if request.action not in env.legal_actions:
        raise HTTPException(status_code=400, detail="Invalid action")
    
    _, reward, done, info = env.step(request.action)
    game["moves"].append(request.action)
    
    return {
        "state": env.to_dict(),
        "reward": reward,
        "done": done,
        "winner": info["winner"],
        "moves": game["moves"]
    }


@app.post("/api/game/ai-battle")
async def start_ai_battle(request: AIBattleRequest):
    """AI 대결 시작"""
    game_id = str(uuid.uuid4())
    env = GomokuEnv(request.board_size)
    env.reset()
    
    # AI 플레이어 생성 (학습되지 않은 랜덤 정책)
    policy_value_fn = PolicyValueFunction(request.board_size)
    mcts = MCTS(num_simulations=request.num_simulations)
    
    games[game_id] = {
        "env": env,
        "moves": [],
        "ai_battle": True,
        "policy_value_fn": policy_value_fn,
        "mcts": mcts,
        "delay_ms": request.delay_ms
    }
    
    return {
        "game_id": game_id,
        "state": env.to_dict()
    }


@app.get("/api/game/{game_id}/ai-move")
async def get_ai_move(game_id: str):
    """AI 착수 수행"""
    if game_id not in games:
        raise HTTPException(status_code=404, detail="Game not found")
    
    game = games[game_id]
    env = game["env"]
    
    done, _ = env.get_done_winner()
    if done:
        return {
            "state": env.to_dict(),
            "done": True,
            "winner": int(env._winner),
            "moves": [int(m) for m in game["moves"]]
        }
    
    # MCTS로 행동 선택
    policy_value_fn = game.get("policy_value_fn")
    mcts = game.get("mcts")
    
    if policy_value_fn is None or mcts is None:
        # AI 플레이어가 없으면 생성
        policy_value_fn = PolicyValueFunction(env.board_size)
        mcts = MCTS(num_simulations=50)
        game["policy_value_fn"] = policy_value_fn
        game["mcts"] = mcts
    
    action, action_probs = mcts.get_action_probs(
        env,
        policy_value_fn,
        temperature=0.5,
        add_noise=False
    )
    
    _, reward, done, info = env.step(int(action))
    game["moves"].append(int(action))
    
    return {
        "action": int(action),
        "row": int(action) // env.board_size,
        "col": int(action) % env.board_size,
        "current_player": int(env.current_player),
        "state": env.to_dict(),
        "done": bool(done),
        "winner": int(info["winner"]),
        "moves": [int(m) for m in game["moves"]]
    }


@app.post("/api/train/start")
async def start_training(request: TrainRequest):
    """AlphaZero 학습 시작 (백그라운드)"""
    global training_status
    
    if training_status["is_training"]:
        raise HTTPException(status_code=400, detail="Training already in progress")
    
    training_status = {
        "is_training": True,
        "current_iteration": 0,
        "total_iterations": request.iterations,
        "current_game": 0,
        "games_per_iteration": request.games_per_iteration,
        "losses": []
    }
    
    def train_worker():
        global training_status
        try:
            trainer = AlphaZeroTrainer(
                board_size=request.board_size,
                num_simulations=request.num_simulations,
                batch_size=32,
                model_save_path=str(PROJECT_ROOT / 'ai' / 'models')
            )
            
            for iteration in range(request.iterations):
                training_status["current_iteration"] = iteration + 1
                
                for game in range(request.games_per_iteration):
                    training_status["current_game"] = game + 1
                    trainer.self_play_game()
                
                for _ in range(request.train_steps):
                    total_loss, _, _ = trainer.train_step()
                    if total_loss > 0:
                        training_status["losses"].append(total_loss)
            
            trainer.save_model("model_web_trained.pt")
            
        finally:
            training_status["is_training"] = False
    
    thread = threading.Thread(target=train_worker)
    thread.start()
    
    return {"message": "Training started", "status": training_status}


@app.get("/api/train/status")
async def get_training_status():
    """학습 상태 조회"""
    return training_status


@app.delete("/api/game/{game_id}")
async def delete_game(game_id: str):
    """게임 삭제"""
    if game_id in games:
        del games[game_id]
    return {"deleted": True}


# 정적 파일 서빙
frontend_path = PROJECT_ROOT / "frontend"
if frontend_path.exists():
    app.mount("/css", StaticFiles(directory=frontend_path / "css"), name="css")
    app.mount("/js", StaticFiles(directory=frontend_path / "js"), name="js")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
