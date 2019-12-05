import matplotlib.pyplot as plt
import numpy as np

from target import Target


class SimGenerator:
    def __init__(self,
                 clutter_poisson,
                 birth_prob,
                 ppt=5,
                 timesteps=200,
                 init_targets=[],
                 region=[(-100, 100), (-100, 100)]):

        self.clutter_poisson = clutter_poisson
        self.birth_prob = birth_prob
        self.ppt = ppt
        self.timesteps = timesteps
        self.init_targets = init_targets
        self.region = region

        self.current_step = 0
        self.last_timestep_targets = self.init_targets
        self.true_targets = {}  # true positions
        self.true_observations = {}  # for true particles
        self.clutter_observations = {}  # for clutter particles
        self.observations = {}  # for true and clutter particles

    def iterate(self, k):
        next_timestep_targets = []
        true_positions = []

        observations = []
        true_observations = []
        clutter_observations = []

        # Update State for Current Targets
        for target in self.last_timestep_targets:
            target.next_state()

            # Check if Target Survives
            current_pos = target.state
            x = current_pos[0, 0]
            y = current_pos[1, 0]
            if x < self.region[0][0] or x > self.region[0][1] \
                    or y < self.region[1][0] or y > self.region[1][1]:
                continue
            next_timestep_targets.append(target)
            true_positions.append((x, y))

            # Create X particles per target
            n_particles = self.ppt
            for p in range(n_particles):
                new_meas = target.get_measurement()
                observations.append(new_meas)
                true_observations.append(new_meas)

        # Generate New Births
        x = np.random.random()
        if x < self.birth_prob:
            x = np.random.uniform(low=self.region[0][0],
                                  high=self.region[0][1])
            y = np.random.uniform(low=self.region[1][0],
                                  high=self.region[1][1])
            new_target = Target(init_state=np.array([[x], [y], [0.0], [0.0]]))
            next_timestep_targets.append(new_target)

            # Create X particles per target
            n_particles = self.ppt
            for p in range(n_particles):
                new_meas = new_target.get_measurement()
                observations.append(new_meas)
                true_observations.append(new_meas)

        # # Generate Clutter and add to observations
        # num_clutter, clutter_pos = self.clutter_model.Sample()
        #
        # if num_clutter > 0:
        #     for i in clutter_pos:
        #         new_observations.append(i)
        #         clutter_observations.append(i)

        # Update Observations and Targets
        self.true_targets[k] = true_positions
        self.true_observations[k] = true_observations
        self.clutter_observations[k] = clutter_observations
        self.observations[k] = observations
        self.last_timestep_targets = next_timestep_targets
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
        self.true_targets = {}
        self.observations = {}

    def plot_iter(self, k, folder='data', show_clutter=False):
        plt.clf()
        plt.xlim([self.region[0][0], self.region[0][1]])
        plt.ylim([self.region[1][0], self.region[1][1]])

        x = []
        y = []
        for t in self.true_targets[k]:
            x.append(t[0])
            y.append(t[1])

        plt.scatter(x, y, label='true_position')

        x_obs = []
        y_obs = []
        for o in self.true_observations[k]:
            x_obs.append(o[0][0])
            y_obs.append(o[1][0])
        plt.scatter(x_obs, y_obs, label='true_observations')

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

    def plot(self, show_clutter=False):
        x = []
        y = []
        for i, truths in self.true_targets.items():
            for t in truths:
                x.append(t[0])
                y.append(t[1])

        plt.scatter(x, y, label='true_position')

        x_obs = []
        y_obs = []
        for i, obs in self.true_observations.items():
            for o in obs:
                x_obs.append(o[0][0])
                y_obs.append(o[1][0])
        plt.scatter(x_obs, y_obs, label='true_observations')

        if show_clutter:
            x_clutter = []
            y_clutter = []
            for i, clutter in self.clutter_observations.items():
                for c in clutter:
                    x_clutter.append(c[0])
                    y_clutter.append(c[1])
            plt.scatter(x_clutter, y_clutter, label='clutter')

        plt.legend()
        plt.savefig('test1.png')
        plt.clf()

    def plot_gif(self, folder='data', show_clutter=False):
        for k in range(0, self.current_step+1):
            self.plot_iter(k, folder=folder, show_clutter=show_clutter)

    def save_data(self):
        pass
