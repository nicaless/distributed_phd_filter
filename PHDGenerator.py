from models import *

class PHDGenerator:
    def __init__(self,
                 timesteps,
                 birth_model,
                 motion_model,
                 observation_model,
                 clutter_model,
                 region=[(-100, 100), (-100, 100)]):

        self.timesteps = timesteps
        self.birth_model = birth_model
        self.motion_model = motion_model
        self.clutter_model = clutter_model
        self.observation_model = observation_model
        self.region = region

        self.surviving_targets = []  # true survivors from last timestep
        self.true_targets = {}  # for true observations
        self.observations = {}  # for true observations and clutter

    def generate(self, k):
        new_observations = []
        next_survivors = []

        # Next timestep Survivors (existing targets that DO NOT leave region)
        for i in self.surviving_targets:
            # Apply Transition Model
            new_pos = self.motion_model.next(self.surviving_targets[i])
            x = new_pos[i][0]
            y = new_pos[i][1]

            if x < self.region[0][0] or x > self.region[0][1] \
                    or y < self.region[1][0] or y > self.region[1][1]:
                continue
            next_survivors.append(new_pos)

            # Apply Observation Model
            new_observations.append(self.observation_model.next(new_pos))

        # Generate New Births
        num_new, new = self.birth_model.birth()

        # Add New Births and their observations to next survivors
        for i in new:
            next_survivors.append(new)
            new_observations.append(self.observation_model.next(i))

        # Generate Clutter and add to observations

        # Update Observations and Targets
        self.true_targets[k] = next_survivors
        self.observations[k] = new_observations


