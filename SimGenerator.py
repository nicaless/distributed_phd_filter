import numpy as np
import pandas as pd

from target import Target


class SimGenerator:
    def __init__(self,
                 clutter_lambda=5,
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
        self.observations = {}  # for true and clutter particles

    def iterate(self, k):
        next_timestep_targets = []

        true_positions = []

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
                draw_state = np.array([[self.region[0][0] + 10],
                                       [self.region[1][0] + 10],
                                       [0.1], [0.1]])
                init_cov = np.diag((0.01, 0.01, 0.01, 0.01))
                sample = np.random.multivariate_normal(draw_state.flat,
                                                       init_cov, size=1)
                init_state = np.array([[sample[0][0]], [sample[0][1]],
                                       [sample[0][2]], [sample[0][3]]])

                new_target = Target(init_state=init_state)
            elif corner == 1:
                draw_state = np.array([[self.region[0][0] + 10],
                                       [self.region[1][1] - 10],
                                       [0.1], [0.1]])
                init_cov = np.diag((0.01, 0.01, 0.01, 0.01))
                sample = np.random.multivariate_normal(draw_state.flat,
                                                           init_cov, size=1)
                init_state = np.array([[sample[0][0]], [sample[0][1]],
                                       [sample[0][2]], [sample[0][3]]])
                new_target = Target(init_state=init_state, dt_2=-1)
            elif corner == 2:
                draw_state = np.array([[self.region[0][1] - 10],
                                       [self.region[1][1] - 10],
                                       [0.1], [0.1]])
                init_cov = np.diag((0.01, 0.01, 0.01, 0.01))
                sample = np.random.multivariate_normal(draw_state.flat,
                                                       init_cov, size=1)
                init_state = np.array([[sample[0][0]], [sample[0][1]],
                                       [sample[0][2]], [sample[0][3]]])
                new_target = Target(init_state=init_state,
                                    dt_1=-1, dt_2=-1)
            else:
                draw_state = np.array([[self.region[0][1] - 10],
                                       [self.region[1][0] + 10],
                                       [0.1], [0.1]])
                init_cov = np.diag((0.01, 0.01, 0.01, 0.01))
                sample = np.random.multivariate_normal(draw_state.flat,
                                                           init_cov, size=1)
                init_state = np.array([[sample[0][0]], [sample[0][1]],
                                       [sample[0][2]], [sample[0][3]]])
                new_target = Target(init_state=init_state, dt_1=-1)

            next_timestep_targets.append(new_target)
            true_positions.append(new_target.get_measurement())

        # Update Observations and Targets
        self.targets[k] = next_timestep_targets
        self.last_timestep_targets = next_timestep_targets

        self.true_positions[k] = true_positions
        self.observations[k] = true_positions
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
        self.observations = {}  # for true and clutter particles

    def save_data(self, path):
        time = []
        x = []
        y = []
        for t, items in self.true_positions.items():
            for item in items:
                time.append(t)
                x.append(item[0][0])
                y.append(item[1][0])
        data = pd.DataFrame([time, x, y])
        data = data.transpose()
        data.columns = ['time', 'x', 'y']
        data.to_csv(path + '/true_positions.csv', index=False)

