import matplotlib.pyplot as plt
import numpy as np

from target import Target


class SimGenerator:
    def __init__(self,
                 clutter_lambda,
                 birth_prob=0.2,
                 birth_poisson_lambda=1,
                 survival_prob=0.95,
                 timesteps=200,
                 init_targets=[],
                 region=[(-50, 50), (-50, 50)]):

        self.clutter_lambda = clutter_lambda
        self.birth_prob = birth_prob
        self.birth_poisson_lambda = birth_poisson_lambda
        self.survival_prob = survival_prob
        self.timesteps = timesteps
        self.init_targets = init_targets
        self.region = region

        self.current_step = 0
        self.last_timestep_targets = self.init_targets
        self.targets = {}
        self.true_positions = {}  # true positions
        self.clutter_observations = {}  # for clutter particles
        self.observations = {}  # for true and clutter particles

    def iterate(self, k):
        next_timestep_targets = []

        true_positions = []
        clutter_observations = []

        # Update State for Current Targets
        for target in self.last_timestep_targets:
            target.next_state()

            # Check if Target Survives
            current_pos = target.state
            x = current_pos[0, 0]
            y = current_pos[1, 0]
            s = np.random.rand()
            if s > self.survival_prob \
                    or x < self.region[0][0] or x > self.region[0][1] \
                    or y < self.region[1][0] or y > self.region[1][1]:
                continue
            next_timestep_targets.append(target)
            true_positions.append(target.get_measurement())

        # Generate New Births
        # Poisson Birth
        num_birth = np.random.poisson(self.birth_poisson_lambda)
        for i in range(num_birth):
            corner = np.random.choice([0, 1, 2, 3])
            if corner == 0:
                new_target = Target(
                    init_state=np.array([
                        [self.region[0][0] + 10 * np.random.randn()],
                        [self.region[1][0] + 10 * np.random.randn()],
                        [0.1], [0.1]]))
            elif corner == 1:
                new_target = Target(
                    init_state=np.array([
                        [self.region[0][0] + 10 * np.random.randn()],
                        [self.region[1][1] + 10 * np.random.randn()],
                        [0.1], [0.1]]),
                    dt_2=-1)
            elif corner == 2:
                new_target = Target(
                    init_state=np.array(
                        [[self.region[0][1] + 10 * np.random.randn()],
                         [self.region[1][1] + 10 * np.random.randn()],
                         [0.1], [0.1]]),
                    dt_1=-1, dt_2=-1)
            else:
                new_target = Target(
                    init_state=np.array(
                        [[self.region[0][1] + 10 * np.random.randn()],
                         [self.region[1][0] + 10 * np.random.randn()],
                         [0.1], [0.1]]),
                    dt_1=-1)

            next_timestep_targets.append(new_target)
            true_positions.append(new_target.get_measurement())

        # Random Uniform Birth
        # x = np.random.random()
        # if x < self.birth_prob:
        #     x = np.random.uniform(low=self.region[0][0],
        #                           high=self.region[0][1])
        #     y = np.random.uniform(low=self.region[1][0],
        #                           high=self.region[1][1])
        #     new_target = Target(init_state=np.array([[x], [y], [0.1], [0.0]]))
        #     next_timestep_targets.append(new_target)
        #     true_positions.append(new_target.get_measurement())

        # num_clutter = np.random.poisson(self.clutter_lambda)
        # TODO: move this to FilterNode object
        num_clutter = self.clutter_lambda
        for i in range(num_clutter):
            x = np.random.uniform(low=self.region[0][0],
                                  high=self.region[0][1])
            y = np.random.uniform(low=self.region[1][0],
                                  high=self.region[1][1])
            clutter_observations.append(np.array([[x], [y]]))

        # Update Observations and Targets
        self.targets[k] = next_timestep_targets
        self.last_timestep_targets = next_timestep_targets

        self.true_positions[k] = true_positions
        self.clutter_observations[k] = clutter_observations
        self.observations[k] = true_positions + clutter_observations
        self.current_step = k

    def generate(self, steps=200, reset=True):
        if reset:
            self.reset()
        steps = min(self.current_step + steps, self.timesteps)
        for i in range(self.current_step, steps):
            self.iterate(i)

    def reset(self):
        self.current_step = 0
        self.last_timestep_targets = self.init_targets
        self.targets = {}
        self.true_positions = {}  # true positions
        self.clutter_observations = {}  # for clutter particles
        self.observations = {}  # for true and clutter particles

    def plot_iter(self, k, folder='data', show_clutter=False):
        plt.clf()
        plt.xlim([self.region[0][0], self.region[0][1]])
        plt.ylim([self.region[1][0], self.region[1][1]])

        x = []
        y = []
        for t in self.true_positions[k]:
            x.append(t[0])
            y.append(t[1])

        plt.scatter(x, y, label='true_position')

        if show_clutter:
            x_clutter = []
            y_clutter = []
            for c in self.clutter_observations[k]:
                x_clutter.append(c[0])
                y_clutter.append(c[1])
            plt.scatter(x_clutter, y_clutter, label='clutter')

        plt.legend()
        plt.savefig('{folder}/{k}.png'.format(folder=folder, k=k))
        plt.clf()

    def plot(self, folder='data', show_clutter=False):
        for k in range(0, self.current_step+1):
            self.plot_iter(k, folder=folder, show_clutter=show_clutter)

    def save_data(self):
        pass
