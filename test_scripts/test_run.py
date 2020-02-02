import numpy as np

from models import Birth, Clutter, Estimate, Measurement, Survival, \
    Transition, TransitionCircle
from PHDFilter import PHDFilter
from PHDGenerator import PHDGenerator

birth_model = Birth(1)
birth_model_1 = Birth(1, region=[(0, 100), (0, 100)])
birth_model_2 = Birth(1, region=[(-100, 0), (-100, 0)])
clutter_model = Clutter(1)
transition_model = Transition()
# transition_model = TransitionCircle()
measurement_model = Measurement(3, .98)
measurement_model_1 = Measurement(3, .98, region=[(0, 100), (0, 100)])
measurement_model_2 = Measurement(3, .98, region=[(-100, 0), (-100, 0)])
survival_model = Survival()
estimation_model = Estimate(1)


generator_1 = PHDGenerator(birth_model=birth_model_1,
                           clutter_model=clutter_model,
                           transition_model=transition_model,
                           measurement_model=measurement_model_1,
                           timesteps=200,
                           region=[(0, 100), (0, 100)])

generator_2 = PHDGenerator(birth_model=birth_model_2,
                           clutter_model=clutter_model,
                           transition_model=transition_model,
                           measurement_model=measurement_model_2,
                           timesteps=200,
                           region=[(-100, 0), (-100, 0)])

generator_1.generate(20)
generator_2.generate(20)

meas_1 = generator_1.observations
meas_2 = generator_2.observations

all_meas = {}
for i in range(20):
    all_meas[i] = []
    for m in meas_1[i]:
        all_meas[i].append(m)
    for m in meas_2[i]:
        all_meas[i].append(m)

filter = PHDFilter(birth_models=[birth_model_1, birth_model_2],
                   clutter_model=clutter_model,
                   measurement_model= measurement_model,
                   transition_model=transition_model,
                   survival_model=survival_model,
                   estimation_model=estimation_model,
                   J=20)

generator_1.plot_gif(folder='test/data1', show_clutter=True)
generator_2.plot_gif(folder='test/data2', show_clutter=True)
# filter.predict()
# filter.update(all_meas[0])
# filter.resample()
# filter.estimate()

filter.step_through(all_meas, folder='test/results')
filter.plot_centroids()




