# YOLO26x CADICA Results

Run: `cadica_selected_seed42_4090`

## Training setup

- Model: `yolo26x.pt`
- Dataset: prepared CADICA selected-keyframe split
- Hardware: Vast.ai `RTX 4090`
- Image size: `512`
- Batch size: `16`
- Workers: `12`
- Optimizer: `AdamW`
- Initial learning rate: `0.001`
- Early stopping patience: `50`

## Outcome

- Training stopped early after `68` epochs
- Total training time: `4306.74s` (`1.197h`)
- Best checkpoint was selected at epoch `18`

## Best checkpoint metrics

Validation metrics for the best checkpoint:

| Metric | Value |
| --- | ---: |
| Precision | 0.27939 |
| Recall | 0.31648 |
| mAP50 | 0.20264 |
| mAP50-95 | 0.07205 |
| Mean IoU | 0.70132 |

Test metrics for the best checkpoint:

| Metric | Value |
| --- | ---: |
| Precision | 0.15825 |
| Recall | 0.15284 |
| mAP50 | 0.10708 |
| mAP50-95 | 0.03668 |
| Mean IoU | 0.68674 |

## IoU support counts

| Split | Images | GT Boxes | Pred Boxes | Matched Pairs |
| --- | ---: | ---: | ---: | ---: |
| Val | 634 | 534 | 251 | 99 |
| Test | 796 | 687 | 112 | 42 |

## Local artifacts

The full run directory, plots, logs, and checkpoints were saved locally at:

- `/Users/iwosmura/projects/angio-demo/Angiography/models/yolo26x/runs/cadica_selected_seed42_4090`

The full checkpoint files were not committed because they are larger than standard GitHub file limits and the repo already ignores `models/yolo26x/runs/`.
