import numpy as np
import matplotlib.pyplot as plt

class kArmedBandit():
    def __init__(self, k):
        self.k = k
        self.mu = np.random.normal(0, 1, size=k)
        self.realised = None
    
    def getReward(self, a):
        assert a >= 0 and a < self.k
        assert self.realised is not None
        return self.realised[a]
    
    def step(self):
        self.realised = np.random.normal(self.mu, 1)
    
    def isOptimalAction(self, a):
        return a in [idx for idx, val in enumerate(self.mu) if val == np.max(self.mu)]
        
class EpsilonLearner():
    def __init__(self, numActions, epsilon):
        self.epsilon = epsilon
        self.numActions = numActions
        self.v = np.zeros(numActions)
        self.n = np.zeros(numActions)
        self.s = np.zeros(numActions)
        self.prev = None
        
    def act(self):
        p = np.random.uniform()
        
        # 1. Explore w.p. epsilon
        if p < self.epsilon:
            self.prev = np.random.randint(self.numActions)
            self.n[self.prev] += 1
            return self.prev
        
        # 2. Exploit w.p. 1 - epsilon
        feasible_set = [idx for idx, val in enumerate(self.v) if val == np.max(self.v)]
        idx = np.random.randint(len(feasible_set))
        self.prev = feasible_set[idx]
        self.n[self.prev] += 1
        return self.prev
    
    def update(self, r):
        assert self.prev is not None
        self.s[self.prev] += r
        self.v[self.prev] = self.s[self.prev]/self.n[self.prev]

class Simulator():
    def __init__(self, environment, learners):
        self.learners = learners
        self.environment = environment
        self.cumulativeReward = [[] for learner in learners]
        self.avgReward = [[] for learner in learners]
        self.optimalAction = [[] for learner in learners]
        self.reward = [[] for learner in learners]
        
    def simulate(self, nRounds):
        for i in range(nRounds):
            self.environment.step()
            for idx, learner in enumerate(self.learners):
                action = learner.act()
                reward = self.environment.getReward(action)
                self.cumulativeReward[idx].append(reward if len(self.cumulativeReward[idx]) == 0 else self.cumulativeReward[idx][-1] + reward)
                self.avgReward[idx].append(self.cumulativeReward[idx][-1]/(i + 1))
                self.reward[idx].append(reward)
                self.optimalAction[idx].append(int(self.environment.isOptimalAction(action)))
                learner.update(reward)
    
    def getCumulativeReward(self):
        return self.cumulativeReward
    
    def getAvgReward(self):
        return self.avgReward
    
    def getPercentOptimalAction(self):
        return self.optimalAction
    
    def getReward(self):
        return self.reward

class ArmedTestBed():
    def __init__(self, nRuns, nRounds, learners, k=10):
        self.nRuns = nRuns
        self.nRounds = nRounds
        self.k = k
        self.experiments = []
        for _ in range(nRuns):
            self.experiments.append(Simulator(kArmedBandit(k), [cls(*args) for cls, args in learners]))
    
    def run(self):
        for i in range(self.nRuns):
            self.experiments[i].simulate(self.nRounds)
    
    def getAvgReward(self):
        return sum([x.getReward() for x in problem.experiments], np.zeros(np.array(problem.experiments[0].getAvgReward()).shape))/self.nRuns
    
    def getPercentOptimalAction(self):
        return sum([x.getPercentOptimalAction() for x in self.experiments], np.zeros(np.array(problem.experiments[0].getAvgReward()).shape))/self.nRuns

if __name__ == "__main__":
    problem = ArmedTestBed(2000, 1000, [
        (EpsilonLearner, (10, 0.00)), # greedy
        (EpsilonLearner, (10, 0.01)), 
        (EpsilonLearner, (10, 0.1))
    ])
    problem.run()
    
    plt.figure(figsize=(15, 10))
    plt.title("Average Reward Plot")
    plt.ylabel("Average Reward")
    plt.xlabel("Steps")
    plt.plot(np.arange(problem.nRounds), problem.getAvgReward()[0], label="ε = 0.0 (Greedy)")
    plt.plot(np.arange(problem.nRounds), problem.getAvgReward()[1], label="ε = 0.01")
    plt.plot(np.arange(problem.nRounds), problem.getAvgReward()[2], label="ε = 0.1")
    plt.legend()
    
    plt.figure(figsize=(15, 10))
    plt.title("Percentage Optimal Action")
    plt.ylabel("% Optimal Action")
    plt.xlabel("Steps")
    plt.plot(np.arange(problem.nRounds), problem.getPercentOptimalAction()[0], label="ε = 0.0 (Greedy)")
    plt.plot(np.arange(problem.nRounds), problem.getPercentOptimalAction()[1], label="ε = 0.01")
    plt.plot(np.arange(problem.nRounds), problem.getPercentOptimalAction()[2], label="ε = 0.1")
    plt.legend()
