from mattergen.scripts.generate import main as generate


generate(
    output_path="new_work/generated_structures/topological_dgf2",
    model_path="new_work/finetuned_models/topological",
    batch_size=10,
    num_batches=1,
    diffusion_guidance_factor=2,
    properties_to_condition_on={"topological": 1},
    record_trajectories=False,
)
