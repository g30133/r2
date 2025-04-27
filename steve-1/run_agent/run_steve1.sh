# Interactive runs in text mode only, you can click on the window to pause and type in a new prompt
python steve1/run_agent/run_steve1.py \
--in_model data/weights/vpt/2x.model \
--in_weights data/weights/steve1/steve1.weights \
--prior_weights data/weights/steve1/steve1_prior.pt \
--cond_scale 6.0 \
