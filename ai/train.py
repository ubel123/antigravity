"""
AlphaZero Training Script
명령줄에서 학습을 실행하기 위한 스크립트
"""
import argparse
import sys
import os

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.alphazero import AlphaZeroTrainer


def main():
    parser = argparse.ArgumentParser(description='AlphaZero Gomoku Training')
    parser.add_argument('--board-size', type=int, default=15, help='Board size (default: 15)')
    parser.add_argument('--iterations', type=int, default=10, help='Number of training iterations')
    parser.add_argument('--games', type=int, default=5, help='Games per iteration')
    parser.add_argument('--train-steps', type=int, default=50, help='Training steps per iteration')
    parser.add_argument('--simulations', type=int, default=100, help='MCTS simulations')
    parser.add_argument('--batch-size', type=int, default=64, help='Training batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--device', type=str, default=None, help='Device (cpu/cuda)')
    parser.add_argument('--model-path', type=str, default='ai/models', help='Model save path')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("AlphaZero Gomoku Training")
    print("=" * 50)
    print(f"Board Size: {args.board_size}x{args.board_size}")
    print(f"Iterations: {args.iterations}")
    print(f"Games per iteration: {args.games}")
    print(f"MCTS Simulations: {args.simulations}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print("=" * 50)
    
    trainer = AlphaZeroTrainer(
        board_size=args.board_size,
        num_simulations=args.simulations,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_save_path=args.model_path,
        device=args.device
    )
    
    trainer.train(
        num_iterations=args.iterations,
        games_per_iteration=args.games,
        train_steps_per_iteration=args.train_steps
    )
    
    print("\nTraining Statistics:")
    stats = trainer.get_stats()
    if stats['total_loss']:
        print(f"  Final Total Loss: {stats['total_loss'][-1]:.4f}")
        print(f"  Average Game Length: {sum(stats['game_lengths']) / len(stats['game_lengths']):.1f}")


if __name__ == '__main__':
    main()
