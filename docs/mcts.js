/**
 * Monte Carlo Tree Search (JavaScript 구현)
 * ONNX 모델과 함께 사용
 */
class MCTSNode {
    constructor(parent = null, priorP = 1.0) {
        this.parent = parent;
        this.children = new Map();
        this.visitCount = 0;
        this.valueSum = 0;
        this.priorP = priorP;
    }

    get value() {
        return this.visitCount === 0 ? 0 : this.valueSum / this.visitCount;
    }

    isLeaf() {
        return this.children.size === 0;
    }

    isRoot() {
        return this.parent === null;
    }

    update(value) {
        this.visitCount++;
        this.valueSum += value;
    }

    updateRecursive(leafValue, negate = true) {
        this.update(leafValue);
        if (!this.isRoot()) {
            this.parent.updateRecursive(negate ? -leafValue : leafValue, negate);
        }
    }

    expand(actionProbs) {
        for (const [action, prob] of actionProbs) {
            if (!this.children.has(action)) {
                this.children.set(action, new MCTSNode(this, prob));
            }
        }
    }
}

class MCTS {
    constructor(numSimulations = 50, pbCBase = 19652, pbCInit = 1.25) {
        this.numSimulations = numSimulations;
        this.pbCBase = pbCBase;
        this.pbCInit = pbCInit;
        this.rootDirichletAlpha = 0.3;
        this.rootNoiseWeight = 0.25;
    }

    async getActionProbs(env, policyValueFn, temperature = 1.0, addNoise = true) {
        const root = new MCTSNode();

        // 루트 노드 확장
        await this.expandLeafNode(root, env, policyValueFn);

        // 노이즈 추가
        if (addNoise) {
            this.addExplorationNoise(root);
        }

        // MCTS 시뮬레이션
        for (let i = 0; i < this.numSimulations; i++) {
            const simEnv = env.copy();
            await this.simulate(root, simEnv, policyValueFn);
        }

        // 방문 횟수 집계
        const actionVisits = [];
        for (let a = 0; a < env.actionSpaceSize; a++) {
            const visits = root.children.has(a) ? root.children.get(a).visitCount : 0;
            actionVisits.push([a, visits]);
        }

        // 확률 계산
        const visits = actionVisits.map(([_, v]) => v);
        let actionProbs;

        if (temperature === 0) {
            actionProbs = new Array(visits.length).fill(0);
            const maxIdx = visits.indexOf(Math.max(...visits));
            actionProbs[maxIdx] = 1;
        } else {
            const visitsTemp = visits.map(v => Math.pow(v, 1 / temperature));
            const total = visitsTemp.reduce((a, b) => a + b, 0);
            actionProbs = total > 0 ? visitsTemp.map(v => v / total) : visits.map(() => 1 / visits.length);
        }

        // 행동 선택
        let action;
        if (temperature === 0) {
            action = visits.indexOf(Math.max(...visits));
        } else {
            action = this.sampleAction(actionProbs);
        }

        return [action, actionProbs];
    }

    sampleAction(probs) {
        const r = Math.random();
        let cumsum = 0;
        for (let i = 0; i < probs.length; i++) {
            cumsum += probs[i];
            if (r < cumsum) return i;
        }
        return probs.length - 1;
    }

    async simulate(node, env, policyValueFn) {
        // Selection
        while (!node.isLeaf()) {
            const [action, child] = this.selectChild(node, env);
            if (action === null) break;
            env.step(action);
            node = child;
        }

        // 게임 종료 체크
        const [done, winner] = env.checkWinner();
        let leafValue;

        if (done) {
            if (winner === 0) leafValue = 0;
            else if (winner === env.currentPlayer) leafValue = 1;
            else leafValue = -1;
        } else {
            leafValue = await this.expandLeafNode(node, env, policyValueFn);
        }

        // Backpropagation
        node.updateRecursive(-leafValue);
    }

    async expandLeafNode(node, env, policyValueFn) {
        const obs = env.getObservation();
        const [actionProbs, value] = await policyValueFn(obs);

        const legalActions = env.legalActions;
        if (legalActions.length === 0) return 0;

        // 유효한 행동만 필터링
        const legalProbs = legalActions.map(a => [a, actionProbs[a]]);
        const totalProb = legalProbs.reduce((sum, [_, p]) => sum + p, 0);
        const normalizedProbs = totalProb > 0
            ? legalProbs.map(([a, p]) => [a, p / totalProb])
            : legalProbs.map(([a, _]) => [a, 1 / legalActions.length]);

        node.expand(normalizedProbs);

        return value;
    }

    selectChild(node, env) {
        const legalActions = env.legalActions;
        let bestScore = -Infinity;
        let bestAction = null;
        let bestChild = node;

        for (const [action, child] of node.children) {
            if (legalActions.includes(action)) {
                const score = this.ucbScore(node, child);
                if (score > bestScore) {
                    bestScore = score;
                    bestAction = action;
                    bestChild = child;
                }
            }
        }

        return [bestAction, bestChild];
    }

    ucbScore(parent, child) {
        const pbC = Math.log((parent.visitCount + this.pbCBase + 1) / this.pbCBase) + this.pbCInit;
        const priorScore = pbC * child.priorP * Math.sqrt(parent.visitCount) / (child.visitCount + 1);
        return priorScore + child.value;
    }

    addExplorationNoise(node) {
        const actions = Array.from(node.children.keys());
        if (actions.length === 0) return;

        const noise = this.dirichlet(actions.length, this.rootDirichletAlpha);

        actions.forEach((action, i) => {
            const child = node.children.get(action);
            child.priorP = (1 - this.rootNoiseWeight) * child.priorP + this.rootNoiseWeight * noise[i];
        });
    }

    dirichlet(n, alpha) {
        // 간단한 Dirichlet 샘플링
        const samples = Array(n).fill(0).map(() => this.gamma(alpha, 1));
        const sum = samples.reduce((a, b) => a + b, 0);
        return samples.map(s => s / sum);
    }

    gamma(alpha, beta) {
        // 간단한 Gamma 분포 샘플링 (Marsaglia and Tsang's method)
        if (alpha < 1) {
            return this.gamma(alpha + 1, beta) * Math.pow(Math.random(), 1 / alpha);
        }
        const d = alpha - 1 / 3;
        const c = 1 / Math.sqrt(9 * d);
        while (true) {
            let x, v;
            do {
                x = this.randn();
                v = 1 + c * x;
            } while (v <= 0);
            v = v * v * v;
            const u = Math.random();
            if (u < 1 - 0.0331 * x * x * x * x) return d * v / beta;
            if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v / beta;
        }
    }

    randn() {
        // Box-Muller 변환
        const u1 = Math.random();
        const u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
}

// 전역 노출
window.MCTS = MCTS;
