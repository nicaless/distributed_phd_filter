from models import Birth, Clutter, Measurement, Transition


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

        self.last_timestep_targets = init_targets
        self.true_targets = {}  # for true observations
        self.observations = {}  # for true observations and clutter

    def generate(self, k):
        new_observations = []
        next_timestep_targets = []

        # Next timestep targets
        for i in range(0, len(self.last_timestep_targets)):
            t = self.last_timestep_targets[i]
            # Apply Transition Model
            new_pos = self.transition_model.AdvanceState(t)
            new_meas = self.measurement_model.Measure(t)

            next_timestep_targets.append(new_pos)
            new_observations.append(new_meas)

        # Generate New Births
        num_births, birth_pos = self.birth_model.Sample()

        # Add New Births and their observations to next survivors
        for i in range(0, num_births):
            t = birth_pos[i]
            next_timestep_targets.append(t)
            birth_meas = self.measurement_model.Measure(t)
            new_observations.append(birth_meas)

        # Generate Clutter and add to observations
        num_clutter, clutter_pos = self.clutter_model.Sample()

        for i in clutter_pos:
            new_observations.append(i)

        # Update Observations and Targets
        self.true_targets[k] = next_timestep_targets
        self.observations[k] = new_observations
        self.last_timestep_targets = next_timestep_targets


