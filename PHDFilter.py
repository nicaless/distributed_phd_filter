from models import *


# TODO: follow from line 550 at https://github.com/hichamjanati/PHD-filter/blob/master/base.py
class PHDFilter:
    def __init__(self, birth, survival,
                 target_motion, measurement,
                 detection, clutter):
        # From models.py
        self.birth = birth
        self.survival = survival
        self.target_motion = target_motion
        self.measurement = measurement

        # floats for now... may turn into models depending on use??
        self.detection = detection
        self.clutter = clutter

        # set of positions of all targets
        self.targets = []

        # set of phds of all targets ?
        self.weights = []

    def predict(self, J):
        # apply birth model
        # TODO: remove this... this will go in generator model?
        num_new_particles, new_particle_positions = self.birth.birth()

        # create weights for newly birthed targets (based on J particles)

        # get new positions for existing targets
        target_positions = self.target_motion.next(self.targets)

        # get new weights for existing particles

        # update self.targets and self.weights with both new and existing

        pass

    def update(self):
        # update weights?
        pass

    # not entirely sure what this step is for?
    def resample(self):
        pass

    # not entirely sure what this step is for?
    def estimate(self):
        pass


