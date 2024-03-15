from matplotlib import pyplot as plt


class TrainingMonitor:
    def __init__(self):
        self.fig, self.ax = plt.subplots(2, 1, figsize=(10, 8))
        self.ax[0].set_title('Training Progress')
        self.ax[0].set_xlabel('Episode')
        self.ax[0].set_ylabel('Reward')
        self.ax[1].set_title('Evaluation Metrics')
        self.ax[1].set_xlabel('Episode')
        self.ax[1].set_ylabel('Average Reward')
        self.training_rewards = []
        self.evaluation_rewards = []

    def update_training_progress(self, episode, reward):
        self.training_rewards.append(reward)
        self.ax[0].plot(self.training_rewards, color='blue')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_evaluation_metrics(self, episode, avg_reward):
        self.evaluation_rewards.append(avg_reward)
        self.ax[1].plot(self.evaluation_rewards, color='green')
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
