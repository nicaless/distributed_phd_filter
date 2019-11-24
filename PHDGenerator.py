import matplotlib.pyplot as plt
import matplotlib.animation as animation


class PHDGenerator:
    def __init__(self,
                 birth_model,
                 clutter_model,
                 transition_model,
                 measurement_model,
                 timesteps=200,
                 init_targets=[],
                 region=[(-100, 100), (-100, 100)]):

        self.birth_model = birth_model
        self.clutter_model = clutter_model
        self.transition_model = transition_model
        self.measurement_model = measurement_model
        self.timesteps = timesteps
        self.region = region

        self.current_step = 0
        self.init_targets = init_targets
        self.last_timestep_targets = self.init_targets
        self.true_targets = {}  # true positions
        self.true_observations = {}  # for true observations
        self.clutter_observations = {}  # for clutter observations
        self.observations = {}  # for true observations and clutter

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
            new_meas = self.measurement_model.Measure(t)

            if new_meas.size > 0:
                new_observations.append(new_meas)
                target_observations.append(new_meas)
                next_timestep_targets.append(new_pos)

        # Generate New Births
        num_births, birth_pos = self.birth_model.Sample()

        # Add New Births and their observations to next survivors
        for i in range(0, num_births):
            t = birth_pos[i]
            birth_meas = self.measurement_model.Measure(t)

            if birth_meas.size > 0:
                new_observations.append(birth_meas)
                target_observations.append(birth_meas)
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

    def plot_iter(self, k, show_clutter=False):
        plt.xlim([-100, 100])
        plt.ylim([-100, 100])

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

        if show_clutter:
            x_clutter = []
            y_clutter = []
            for c in self.clutter_observations[k]:
                x_clutter.append(c[0])
                y_clutter.append(c[1])
            plt.scatter(x_clutter, y_clutter, label='clutter')

        plt.legend()
        plt.savefig('test/{k}.png'.format(k=k))
        plt.clf()

    def plot(self, show_clutter=False):
        x = []
        y = []
        for i, truths in self.true_targets.items():
            for t in truths:
                x.append(t[0][0])
                y.append(t[1][0])

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
                    print(c)
                    x_clutter.append(c[0])
                    y_clutter.append(c[1])
            plt.scatter(x_clutter, y_clutter, label='clutter')

        plt.legend()
        plt.savefig('test1.png')

    def plot_gif(self, show_clutter=False):
        for k in range(0, self.current_step+1):
            self.plot_iter(k, show_clutter=show_clutter)







