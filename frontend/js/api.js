/**
 * API 통신 모듈
 */
const API_BASE = '';

const api = {
    /**
     * 새 게임 생성
     */
    async newGame(boardSize = 15) {
        const response = await fetch(`${API_BASE}/api/game/new`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ board_size: boardSize })
        });
        return response.json();
    },

    /**
     * 게임 상태 조회
     */
    async getGame(gameId) {
        const response = await fetch(`${API_BASE}/api/game/${gameId}`);
        return response.json();
    },

    /**
     * 수동 착수
     */
    async makeMove(gameId, action) {
        const response = await fetch(`${API_BASE}/api/game/${gameId}/move`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action })
        });
        return response.json();
    },

    /**
     * AI 대결 시작
     */
    async startAIBattle(boardSize, numSimulations, delayMs) {
        const response = await fetch(`${API_BASE}/api/game/ai-battle`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                board_size: boardSize,
                num_simulations: numSimulations,
                delay_ms: delayMs
            })
        });
        return response.json();
    },

    /**
     * AI 착수 수행
     */
    async getAIMove(gameId) {
        const response = await fetch(`${API_BASE}/api/game/${gameId}/ai-move`);
        return response.json();
    },

    /**
     * 학습 시작
     */
    async startTraining(config) {
        const response = await fetch(`${API_BASE}/api/train/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(config)
        });
        return response.json();
    },

    /**
     * 학습 상태 조회
     */
    async getTrainingStatus() {
        const response = await fetch(`${API_BASE}/api/train/status`);
        return response.json();
    },

    /**
     * 게임 삭제
     */
    async deleteGame(gameId) {
        const response = await fetch(`${API_BASE}/api/game/${gameId}`, {
            method: 'DELETE'
        });
        return response.json();
    }
};

// 전역 노출
window.api = api;
