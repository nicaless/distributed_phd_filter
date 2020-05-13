import matplotlib.pyplot as plt


class PHDGenerator:
    def __init__(self,
                 birth_model,
                 clutter_model,
                 transition_model,
                 measurement_model,
                 timesteps=200,
                 init_targets=[],
                 region=[(-100, 100), (-100, 100)],
                 ppt=1):

        self.birth_model = birth_model
        self.clutter_model = clutter_model
        self.transition_model = transition_model
        self.measurement_model = measurement_model
        self.timesteps = timesteps
        self.region = region
        self.ppt = ppt
        # TODO: set region for birth model, clutter model, measurement model(?)

        self.current_step = 0
        self.init_targets = init_targets
        self.last_timestep_targets = self.init_targets
        self.true_targets = {}  # true positions
        self.birth_targets = {}  # new births
        self.true_observations = {}  # for true particles
        self.clutter_observations = {}  # for clutter particles
        self.observations = {}  # for true and clutter particles

    def iterate(self, k):
        new_observations = []
        target_observations = []
        clutter_observations = []
        next_timestep_targets = []

        # Next timestep targets
        for i in range(0, len(self.last_timestep_targets)):
            t = self.last_timestep_targets[i]
            # Apply Transition Model
            new_pos = self.transition_model.AdvanceState(t)
            # Create X particles per target
            n_particles = self.ppt
            for p in range(n_particles):
                new_meas = self.measurement_model.Measure(t)

                if new_meas.size > 0:
                    new_observations.append(new_meas)
                    target_observations.append(new_meas)
            next_timestep_targets.append(new_pos)

        # Generate New Births
        num_births, birth_pos = self.birth_model.Sample()
        self.birth_targets[k] = []

        # Add New Births and their observations to next survivors
        for i in range(0, num_births):
            t = birth_pos[i]
            # Create X particles per target
            n_particles = self.ppt
            for p in range(n_particles):
                birth_meas = self.measurement_model.Measure(t)

                if birth_meas.size > 0:
                    new_observations.append(birth_meas)
                    target_observations.append(birth_meas)
                # self.birth_targets[k].append(birth_meas)
            self.birth_targets[k].append(t)
            next_timestep_targets.append(t)

        # Generate Clutter and add to observations
        num_clutter, clutter_pos = self.clutter_model.Sample()

        if num_clutter > 0:
            for i in clutter_pos:
                new_observations.append(i)
                clutter_observations.append(i)

        # Update Observations and Targets
        self.true_targets[k] = next_timestep_targets
        self.true_observations[k] = target_observations
        self.clutter_observations[k] = clutter_observations
        self.observations[k] = new_observations
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
        plt.xlim([self.region[0][0], self.region[0][1]])
        plt.ylim([self.region[1][0], self.region[1][1]])
        # plt.xlim([-10, 10])
        # plt.ylim([-10, 10])

        x = []
        y = []
        for t in self.true_targets[k]:
            x.append(t[0][0])
            y.append(t[1][0])

        plt.scatter(x, y, label='true_position')

        x_obs = []
        y_obs = []
        for o in self.true_observations[k]:
            x_obs.append(o[0][0])
            y_obs.append(o[1][0])
        plt.scatter(x_obs, y_obs, label='true_observations')

        x = []
        y = []
        for b in self.birth_targets[k]:
            x.append(b[0][0])
            y.append(b[1][0])

        plt.scatter(x, y, label='new_births')

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
                x.append(t[0][0])
                y.append(t[1][0])

        plt.scatter(x, y, label='true_position')

        x = []
        y = []
        for i, births in self.birth_targets.items():
            for b in births:
                x.append(b[0][0])
                y.append(b[1][0])

        plt.scatter(x, y, label='new_births')

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

    def plot_gif(self, folder='data', show_clutter=False):
        for k in range(0, self.current_step+1):
            self.plot_iter(k, folder=folder, show_clutter=show_clutter)

    def save_data(self):
        pass







