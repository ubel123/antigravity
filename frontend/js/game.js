/**
 * Gomoku Game - Canvas 렌더링 및 AI 대결 로직
 */
class GomokuGame {
    constructor() {
        this.canvas = document.getElementById('game-board');
        this.ctx = this.canvas.getContext('2d');
        this.boardSize = 15;
        this.cellSize = 0;
        this.padding = 30;
        this.gameId = null;
        this.board = [];
        this.moves = [];
        this.moveCount = 0;
        this.lastAction = null;
        this.isRunning = false;
        this.delayMs = 500;

        this.init();
        this.bindEvents();
    }

    init() {
        this.calculateSizes();
        this.drawBoard();
    }

    calculateSizes() {
        const canvasSize = this.canvas.width - 2 * this.padding;
        this.cellSize = canvasSize / (this.boardSize - 1);
    }

    bindEvents() {
        // AI 대결 시작
        document.getElementById('btn-start-battle').addEventListener('click', () => {
            this.startAIBattle();
        });

        // 중지
        document.getElementById('btn-stop').addEventListener('click', () => {
            this.stopGame();
        });

        // 학습 시작
        document.getElementById('btn-start-train').addEventListener('click', () => {
            this.startTraining();
        });

        // 보드 크기 변경
        document.getElementById('board-size').addEventListener('change', (e) => {
            this.boardSize = parseInt(e.target.value);
            this.calculateSizes();
            this.resetBoard();
        });
    }

    resetBoard() {
        this.board = Array(this.boardSize).fill(null).map(() =>
            Array(this.boardSize).fill(0)
        );
        this.moves = [];
        this.moveCount = 0;
        this.lastAction = null;
        document.getElementById('move-count').textContent = '0';
        document.getElementById('board-overlay').classList.remove('active');
        this.drawBoard();
    }

    drawBoard() {
        const ctx = this.ctx;
        const canvas = this.canvas;

        // 배경
        ctx.fillStyle = '#c9a227';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // 나무 질감 효과
        ctx.fillStyle = 'rgba(139, 115, 85, 0.1)';
        for (let i = 0; i < canvas.height; i += 3) {
            ctx.fillRect(0, i, canvas.width, 1);
        }

        // 격자선
        ctx.strokeStyle = '#8b7355';
        ctx.lineWidth = 1;

        for (let i = 0; i < this.boardSize; i++) {
            const pos = this.padding + i * this.cellSize;

            // 가로선
            ctx.beginPath();
            ctx.moveTo(this.padding, pos);
            ctx.lineTo(canvas.width - this.padding, pos);
            ctx.stroke();

            // 세로선
            ctx.beginPath();
            ctx.moveTo(pos, this.padding);
            ctx.lineTo(pos, canvas.height - this.padding);
            ctx.stroke();
        }

        // 화점 (star points)
        this.drawStarPoints();

        // 돌 그리기
        for (let i = 0; i < this.boardSize; i++) {
            for (let j = 0; j < this.boardSize; j++) {
                if (this.board[i] && this.board[i][j] !== 0) {
                    this.drawStone(i, j, this.board[i][j]);
                }
            }
        }

        // 마지막 착수 표시
        if (this.lastAction !== undefined && this.lastAction !== null) {
            const row = Math.floor(this.lastAction / this.boardSize);
            const col = this.lastAction % this.boardSize;
            if (row >= 0 && col >= 0 && row < this.boardSize && col < this.boardSize) {
                this.drawLastMoveMarker(row, col);
            }
        }
    }

    drawStarPoints() {
        const ctx = this.ctx;
        ctx.fillStyle = '#5a4a3a';

        let points = [];
        if (this.boardSize === 15) {
            points = [[3, 3], [3, 7], [3, 11], [7, 3], [7, 7], [7, 11], [11, 3], [11, 7], [11, 11]];
        } else if (this.boardSize === 13) {
            points = [[3, 3], [3, 6], [3, 9], [6, 6], [9, 3], [9, 6], [9, 9]];
        } else if (this.boardSize === 9) {
            points = [[2, 2], [2, 6], [4, 4], [6, 2], [6, 6]];
        }

        points.forEach(([row, col]) => {
            const x = this.padding + col * this.cellSize;
            const y = this.padding + row * this.cellSize;
            ctx.beginPath();
            ctx.arc(x, y, 4, 0, Math.PI * 2);
            ctx.fill();
        });
    }

    drawStone(row, col, player) {
        const ctx = this.ctx;
        const x = this.padding + col * this.cellSize;
        const y = this.padding + row * this.cellSize;
        const radius = this.cellSize * 0.43;

        // 그림자
        ctx.beginPath();
        ctx.arc(x + 2, y + 2, radius, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
        ctx.fill();

        // 돌
        const gradient = ctx.createRadialGradient(
            x - radius * 0.3, y - radius * 0.3, 0,
            x, y, radius
        );

        if (player === 1) {
            // 흑돌
            gradient.addColorStop(0, '#4a4a4a');
            gradient.addColorStop(1, '#1a1a1a');
        } else {
            // 백돌
            gradient.addColorStop(0, '#ffffff');
            gradient.addColorStop(1, '#d0d0d0');
        }

        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.fill();
    }

    drawLastMoveMarker(row, col) {
        const ctx = this.ctx;
        const x = this.padding + col * this.cellSize;
        const y = this.padding + row * this.cellSize;
        const player = this.board[row][col];

        ctx.beginPath();
        ctx.arc(x, y, this.cellSize * 0.15, 0, Math.PI * 2);
        ctx.fillStyle = player === 1 ? '#ff6b6b' : '#ff6b6b';
        ctx.fill();
    }

    async startAIBattle() {
        const boardSize = parseInt(document.getElementById('board-size').value);
        const simulations = parseInt(document.getElementById('simulations').value);
        this.delayMs = parseInt(document.getElementById('delay').value);

        this.boardSize = boardSize;
        this.calculateSizes();
        this.resetBoard();

        this.log('AI 대결 시작...');
        this.updateButtons(true);

        try {
            const response = await api.startAIBattle(boardSize, simulations, this.delayMs);
            this.gameId = response.game_id;
            this.isRunning = true;

            this.updateBoardFromState(response.state);
            this.runAIBattle();

        } catch (error) {
            this.log('오류: ' + error.message, 'error');
            this.updateButtons(false);
        }
    }

    async runAIBattle() {
        while (this.isRunning) {
            try {
                // 현재 플레이어 표시
                const currentPlayer = this.getCurrentPlayer();
                this.updatePlayerStatus(currentPlayer);

                // AI 착수
                const response = await api.getAIMove(this.gameId);

                if (response.action !== undefined && response.action !== null) {
                    this.lastAction = response.action;
                    this.updateBoardFromState(response.state);
                    this.moveCount++;

                    const playerName = response.current_player === 1 ? '흑' : '백';
                    this.log(`${playerName}: (${response.row}, ${response.col})`,
                        response.current_player === 1 ? 'black' : 'white');
                }

                document.getElementById('move-count').textContent = this.moveCount;

                if (response.done) {
                    this.isRunning = false;
                    this.showGameResult(response.winner);
                    break;
                }

                // 딜레이
                await this.delay(this.delayMs);

            } catch (error) {
                this.log('오류: ' + error.message, 'error');
                this.isRunning = false;
                break;
            }
        }

        this.updateButtons(false);
    }

    updateBoardFromState(state) {
        this.board = state.board;
        this.moves = state.legal_actions ?
            Array(this.boardSize * this.boardSize)
                .fill(0)
                .map((_, i) => i)
                .filter(i => !state.legal_actions.includes(i)) :
            [];

        // moves 수정: 실제 착수된 수 계산
        let moveCount = 0;
        for (let i = 0; i < this.boardSize; i++) {
            for (let j = 0; j < this.boardSize; j++) {
                if (this.board[i][j] !== 0) moveCount++;
            }
        }
        this.moves = new Array(moveCount);

        this.drawBoard();
    }

    getCurrentPlayer() {
        let count = 0;
        for (let i = 0; i < this.boardSize; i++) {
            for (let j = 0; j < this.boardSize; j++) {
                if (this.board[i] && this.board[i][j] !== 0) count++;
            }
        }
        return (count % 2 === 0) ? 1 : 2;
    }

    updatePlayerStatus(currentPlayer) {
        const status1 = document.getElementById('player1-status');
        const status2 = document.getElementById('player2-status');

        if (currentPlayer === 1) {
            status1.textContent = '생각중...';
            status1.classList.add('thinking');
            status2.textContent = '대기중';
            status2.classList.remove('thinking');
        } else {
            status2.textContent = '생각중...';
            status2.classList.add('thinking');
            status1.textContent = '대기중';
            status1.classList.remove('thinking');
        }
    }

    showGameResult(winner) {
        const overlay = document.getElementById('board-overlay');
        const winnerText = document.getElementById('winner-text');

        if (winner === 0) {
            winnerText.textContent = '무승부!';
            this.log('게임 종료: 무승부', 'win');
        } else if (winner === 1) {
            winnerText.textContent = '🏆 흑 승리!';
            this.log('게임 종료: 흑 승리!', 'win');
        } else {
            winnerText.textContent = '🏆 백 승리!';
            this.log('게임 종료: 백 승리!', 'win');
        }

        overlay.classList.add('active');

        document.getElementById('player1-status').textContent = winner === 1 ? '승리!' : '패배';
        document.getElementById('player2-status').textContent = winner === 2 ? '승리!' : '패배';
        document.getElementById('player1-status').classList.remove('thinking');
        document.getElementById('player2-status').classList.remove('thinking');
    }

    stopGame() {
        this.isRunning = false;
        this.log('게임 중지됨');
        this.updateButtons(false);
    }

    updateButtons(isRunning) {
        document.getElementById('btn-start-battle').disabled = isRunning;
        document.getElementById('btn-stop').disabled = !isRunning;
    }

    async startTraining() {
        const iterations = parseInt(document.getElementById('train-iterations').value);
        const games = parseInt(document.getElementById('train-games').value);
        const boardSize = parseInt(document.getElementById('board-size').value);

        this.log('AlphaZero 학습 시작...');

        try {
            const response = await api.startTraining({
                board_size: Math.min(boardSize, 9), // 학습 시 작은 보드 권장
                iterations: iterations,
                games_per_iteration: games,
                train_steps: 30,
                num_simulations: 50
            });

            this.log('학습이 백그라운드에서 진행중...');
            this.pollTrainingStatus();

        } catch (error) {
            this.log('학습 오류: ' + error.message, 'error');
        }
    }

    async pollTrainingStatus() {
        const statusEl = document.querySelector('#training-status .status-text');
        const progressEl = document.getElementById('training-progress');

        const poll = async () => {
            try {
                const status = await api.getTrainingStatus();

                if (status.is_training) {
                    const progress = (status.current_iteration / status.total_iterations) * 100;
                    progressEl.style.width = progress + '%';
                    statusEl.textContent = `학습중: ${status.current_iteration}/${status.total_iterations} 반복, 게임 ${status.current_game}/${status.games_per_iteration}`;

                    setTimeout(poll, 2000);
                } else {
                    progressEl.style.width = '100%';
                    statusEl.textContent = '학습 완료!';
                    this.log('AlphaZero 학습 완료!', 'win');
                }
            } catch (error) {
                statusEl.textContent = '상태 확인 실패';
            }
        };

        poll();
    }

    log(message, type = '') {
        const logContainer = document.getElementById('game-log');
        const entry = document.createElement('div');
        entry.className = 'log-entry ' + type;
        entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        logContainer.insertBefore(entry, logContainer.firstChild);

        // 최대 50개 로그 유지
        while (logContainer.children.length > 50) {
            logContainer.removeChild(logContainer.lastChild);
        }
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// 앱 시작
document.addEventListener('DOMContentLoaded', () => {
    window.game = new GomokuGame();
});
