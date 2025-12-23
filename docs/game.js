/**
 * Game UI - Canvas 렌더링 및 ONNX 모델 통합
 */
class GomokuGame {
    constructor() {
        this.canvas = document.getElementById('game-board');
        this.ctx = this.canvas.getContext('2d');
        this.boardSize = 9;
        this.cellSize = 0;
        this.padding = 25;
        this.env = null;
        this.session = null;
        this.mcts = null;
        this.isRunning = false;
        this.moveCount = 0;
        this.lastAction = null;
        this.delayMs = 300;

        this.init();
    }

    async init() {
        this.calculateSizes();
        this.drawBoard();
        this.bindEvents();
        await this.loadModel();
    }

    async loadModel() {
        const statusEl = document.getElementById('model-status');
        try {
            statusEl.textContent = '모델 로딩 중...';

            // ONNX Runtime Web WASM 파일 경로 설정 (CDN 1.17.0)
            ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/";
            ort.env.wasm.numThreads = 1;

            const options = {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            };

            // 모델 로드 (단일 .onnx 파일)
            // fetch 후 ArrayBuffer로 넘기는 것이 GitHub Pages에서 가장 안정적임
            const response = await fetch('./model.onnx');
            if (!response.ok) throw new Error(`Model Load Failed: ${response.status}`);

            const buffer = await response.arrayBuffer();
            this.session = await ort.InferenceSession.create(buffer, options);

            statusEl.textContent = '✓ 모델 로드 완료!';
            statusEl.classList.add('loaded');
            this.log('ONNX 모델 로드 완료');
        } catch (error) {
            statusEl.textContent = '✗ 모델 로드 실패: ' + error.message;
            statusEl.classList.add('error');
            this.log('모델 로드 실패: ' + error.message, 'error');
            console.error(error);
        }
    }

    calculateSizes() {
        const canvasSize = this.canvas.width - 2 * this.padding;
        this.cellSize = canvasSize / (this.boardSize - 1);
    }

    bindEvents() {
        document.getElementById('btn-start-battle').addEventListener('click', () => {
            this.startAIBattle();
        });

        document.getElementById('btn-stop').addEventListener('click', () => {
            this.stopGame();
        });

        document.getElementById('btn-restart').addEventListener('click', () => {
            document.getElementById('board-overlay').classList.remove('active');
            this.resetBoard();
        });
    }

    resetBoard() {
        this.env = new GomokuEnv(this.boardSize);
        this.env.reset();
        this.moveCount = 0;
        this.lastAction = null;
        document.getElementById('move-count').textContent = '0';
        document.getElementById('board-overlay').classList.remove('active');
        document.getElementById('player1-status').textContent = '대기중';
        document.getElementById('player2-status').textContent = '대기중';
        this.drawBoard();
    }

    drawBoard() {
        const ctx = this.ctx;
        const canvas = this.canvas;

        // 배경
        ctx.fillStyle = '#c9a227';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // 나무 질감
        ctx.fillStyle = 'rgba(139, 115, 85, 0.1)';
        for (let i = 0; i < canvas.height; i += 3) {
            ctx.fillRect(0, i, canvas.width, 1);
        }

        // 격자선
        ctx.strokeStyle = '#8b7355';
        ctx.lineWidth = 1;

        for (let i = 0; i < this.boardSize; i++) {
            const pos = this.padding + i * this.cellSize;
            ctx.beginPath();
            ctx.moveTo(this.padding, pos);
            ctx.lineTo(canvas.width - this.padding, pos);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(pos, this.padding);
            ctx.lineTo(pos, canvas.height - this.padding);
            ctx.stroke();
        }

        // 화점
        this.drawStarPoints();

        // 돌 그리기
        if (this.env) {
            for (let i = 0; i < this.boardSize; i++) {
                for (let j = 0; j < this.boardSize; j++) {
                    if (this.env.board[i][j] !== 0) {
                        this.drawStone(i, j, this.env.board[i][j]);
                    }
                }
            }
        }

        // 마지막 착수 표시
        if (this.lastAction !== null) {
            const row = Math.floor(this.lastAction / this.boardSize);
            const col = this.lastAction % this.boardSize;
            this.drawLastMoveMarker(row, col);
        }
    }

    drawStarPoints() {
        const ctx = this.ctx;
        ctx.fillStyle = '#5a4a3a';
        const points = [[2, 2], [2, 6], [4, 4], [6, 2], [6, 6]];

        points.forEach(([row, col]) => {
            const x = this.padding + col * this.cellSize;
            const y = this.padding + row * this.cellSize;
            ctx.beginPath();
            ctx.arc(x, y, 3, 0, Math.PI * 2);
            ctx.fill();
        });
    }

    drawStone(row, col, player) {
        const ctx = this.ctx;
        const x = this.padding + col * this.cellSize;
        const y = this.padding + row * this.cellSize;
        const radius = this.cellSize * 0.42;

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
            gradient.addColorStop(0, '#4a4a4a');
            gradient.addColorStop(1, '#1a1a1a');
        } else {
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

        ctx.beginPath();
        ctx.arc(x, y, this.cellSize * 0.12, 0, Math.PI * 2);
        ctx.fillStyle = '#ff6b6b';
        ctx.fill();
    }

    async policyValueFn(obs) {
        if (!this.session) {
            // 모델이 없으면 균등 분포 반환
            const policy = new Float32Array(this.boardSize * this.boardSize).fill(1 / (this.boardSize * this.boardSize));
            return [policy, 0];
        }

        const tensor = new ort.Tensor('float32', obs, [1, 3, this.boardSize, this.boardSize]);
        const results = await this.session.run({ input: tensor });

        const policy = results.policy.data;
        const value = results.value.data[0];

        return [policy, value];
    }

    async startAIBattle() {
        const simulations = parseInt(document.getElementById('simulations').value);
        this.delayMs = parseInt(document.getElementById('delay').value);

        this.resetBoard();
        this.mcts = new MCTS(simulations);
        this.isRunning = true;

        this.log('AI 대결 시작...');
        this.updateButtons(true);

        await this.runAIBattle();
    }

    async runAIBattle() {
        while (this.isRunning && !this.env.done) {
            try {
                // 현재 플레이어 표시
                this.updatePlayerStatus(this.env.currentPlayer);

                // AI 착수
                const [action, _] = await this.mcts.getActionProbs(
                    this.env,
                    (obs) => this.policyValueFn(obs),
                    0.5,
                    true
                );

                const row = Math.floor(action / this.boardSize);
                const col = action % this.boardSize;

                this.env.step(action);
                this.lastAction = action;
                this.moveCount++;

                document.getElementById('move-count').textContent = this.moveCount;
                this.drawBoard();

                const playerName = this.env.currentPlayer === 1 ? '백' : '흑'; // 착수한 플레이어
                this.log(`${playerName}: (${row}, ${col})`, playerName === '흑' ? 'black' : 'white');

                if (this.env.done) {
                    this.showGameResult(this.env.winner);
                    break;
                }

                await this.delay(this.delayMs);

            } catch (error) {
                this.log('오류: ' + error.message, 'error');
                console.error(error);
                this.isRunning = false;
                break;
            }
        }

        this.updateButtons(false);
    }

    updatePlayerStatus(currentPlayer) {
        const status1 = document.getElementById('player1-status');
        const status2 = document.getElementById('player2-status');
        const card1 = document.getElementById('player1-card');
        const card2 = document.getElementById('player2-card');

        if (currentPlayer === 1) {
            status1.textContent = '생각중...';
            status1.classList.add('thinking');
            status2.textContent = '대기중';
            status2.classList.remove('thinking');
            card1.classList.add('active');
            card2.classList.remove('active');
        } else {
            status2.textContent = '생각중...';
            status2.classList.add('thinking');
            status1.textContent = '대기중';
            status1.classList.remove('thinking');
            card2.classList.add('active');
            card1.classList.remove('active');
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

        document.getElementById('player1-status').textContent = winner === 1 ? '승리!' : (winner === 0 ? '무승부' : '패배');
        document.getElementById('player2-status').textContent = winner === 2 ? '승리!' : (winner === 0 ? '무승부' : '패배');
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

    log(message, type = '') {
        const logContainer = document.getElementById('game-log');
        const entry = document.createElement('div');
        entry.className = 'log-entry ' + type;
        entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
        logContainer.insertBefore(entry, logContainer.firstChild);

        while (logContainer.children.length > 30) {
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
