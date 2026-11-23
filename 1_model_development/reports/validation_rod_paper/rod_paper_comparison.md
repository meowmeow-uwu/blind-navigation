# ROD Models: Test Set Validation vs. Paper Baseline

- **Evaluated at**: 2026-11-09 16:36:50
- **Test Dataset**: `/home/dutai/pbl6-blind-navigation/blind-navigation/1_model_development/datasets/raw/ROD-Dataset/dataset/test`
- **Image Size**: 640

| Model | Metric | Measured (Local) | Paper / Checkpoint | Difference |
| :--- | :--- | :--- | :--- | :--- |
| **yolov8n** | mAP@0.50 | 0.9104 | 0.8757 | ++0.0347 |
|  | mAP@50-95 | 0.7430 | 0.7184 | ++0.0246 |
|  | Precision | 0.9094 | 0.9084 | ++0.0010 |
|  | Recall | 0.8679 | 0.8699 | -0.0019 |
|  | Params | 3015723 | 3010523 | ++5200.0000 |
|  | FPS | 461.6800 | 79.7200 | ++381.9600 |
| **yolov9t** | mAP@0.50 | 0.9195 | 0.8825 | ++0.0370 |
|  | mAP@50-95 | 0.7727 | 0.7496 | ++0.0231 |
|  | Precision | 0.9015 | 0.9042 | -0.0028 |
|  | Recall | 0.8830 | 0.8800 | ++0.0030 |
|  | Params | 2010283 | 1975659 | ++34624.0000 |
|  | FPS | 393.0100 | 70.4400 | ++322.5700 |
| **yolov12n** | mAP@0.50 | 0.9111 | 0.8777 | ++0.0334 |
|  | mAP@50-95 | 0.7610 | 0.7386 | ++0.0224 |
|  | Precision | 0.8967 | 0.8990 | -0.0023 |
|  | Recall | 0.8874 | 0.8893 | -0.0019 |
|  | Params | 2572923 | 2561603 | ++11320.0000 |
|  | FPS | 366.4600 | 59.7300 | ++306.7300 |
| **yolov26n** | mAP@0.50 | 0.9165 | 0.8737 | ++0.0427 |
|  | mAP@50-95 | 0.7715 | 0.7422 | ++0.0293 |
|  | Precision | 0.9091 | 0.9246 | -0.0156 |
|  | Recall | 0.8907 | 0.8756 | ++0.0152 |
|  | Params | 2513550 | 2379711 | ++133839.0000 |
|  | FPS | 456.4300 | 77.6100 | ++378.8200 |

