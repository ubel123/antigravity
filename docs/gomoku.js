/**
 * Gomoku 게임 환경 (JavaScript 구현)
 */
class GomokuEnv {
    constructor(boardSize = 9) {
        this.boardSize = boardSize;
        this.actionSpaceSize = boardSize * boardSize;
        this.reset();
    }

    reset() {
        this.board = Array(this.boardSize).fill(null).map(() =>
            Array(this.boardSize).fill(0)
        );
        this.currentPlayer = 1; // 1: 흑, 2: 백
        this.done = false;
        this.winner = 0;
        return this.getObservation();
    }

    get legalActions() {
        const actions = [];
        for (let i = 0; i < this.boardSize; i++) {
            for (let j = 0; j < this.boardSize; j++) {
                if (this.board[i][j] === 0) {
                    actions.push(i * this.boardSize + j);
                }
            }
        }
        return actions;
    }

    step(action) {
        if (this.done) {
            return { obs: this.getObservation(), reward: 0, done: true, winner: this.winner };
        }

        const row = Math.floor(action / this.boardSize);
        const col = action % this.boardSize;

        if (this.board[row][col] !== 0) {
            throw new Error(`Invalid action: position (${row}, ${col}) is occupied`);
        }

        this.board[row][col] = this.currentPlayer;

        const [done, winner] = this.checkWinner();
        this.done = done;
        this.winner = winner;

        let reward = 0;
        if (done) {
            if (winner === this.currentPlayer) reward = 1;
            else if (winner !== 0) reward = -1;
        }

        if (!done) {
            this.currentPlayer = 3 - this.currentPlayer;
        }

        return { obs: this.getObservation(), reward, done, winner };
    }

    checkWinner() {
        const directions = [[0, 1], [1, 0], [1, 1], [1, -1]];

        for (let i = 0; i < this.boardSize; i++) {
            for (let j = 0; j < this.boardSize; j++) {
                if (this.board[i][j] === 0) continue;

                const player = this.board[i][j];

                for (const [di, dj] of directions) {
                    let count = 1;
                    let ni = i + di, nj = j + dj;

                    while (ni >= 0 && ni < this.boardSize &&
                        nj >= 0 && nj < this.boardSize &&
                        this.board[ni][nj] === player) {
                        count++;
                        ni += di;
                        nj += dj;
                    }

                    if (count >= 5) {
                        return [true, player];
                    }
                }
            }
        }

        if (this.legalActions.length === 0) {
            return [true, 0];
        }

        return [false, 0];
    }

    getObservation() {
        // 3채널 관측값: [현재 플레이어 돌, 상대 돌, 현재 플레이어 표시]
        const obs = new Float32Array(3 * this.boardSize * this.boardSize);
        const size = this.boardSize * this.boardSize;

        for (let i = 0; i < this.boardSize; i++) {
            for (let j = 0; j < this.boardSize; j++) {
                const idx = i * this.boardSize + j;
                // 채널 0: 현재 플레이어의 돌
                obs[idx] = this.board[i][j] === this.currentPlayer ? 1 : 0;
                // 채널 1: 상대 플레이어의 돌
                obs[size + idx] = this.board[i][j] === (3 - this.currentPlayer) ? 1 : 0;
                // 채널 2: 현재 플레이어 표시
                obs[2 * size + idx] = this.currentPlayer === 1 ? 1 : 0;
            }
        }

        return obs;
    }

    copy() {
        const newEnv = new GomokuEnv(this.boardSize);
        newEnv.board = this.board.map(row => [...row]);
        newEnv.currentPlayer = this.currentPlayer;
        newEnv.done = this.done;
        newEnv.winner = this.winner;
        return newEnv;
    }
}

// 전역 노출
window.GomokuEnv = GomokuEnv;
