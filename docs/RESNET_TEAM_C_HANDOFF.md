# Team C handoff

Checkpoints: `runs/resnet18_CT_<task>_<method>_pre_seed<seed>/best.pt`; weights must be shared separately.
Use the shared model builder with checkpoint `args` and `state_dict`, and `gradcam_target_layer` for Grad-CAM.
Keep Team A’s ROI preprocessing and the checkpoint’s validation-selected thresholds unchanged.
