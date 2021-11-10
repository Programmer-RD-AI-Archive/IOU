def create_iou(self, preds: torch.tensor, targets: torch.tensor) -> float:
        ious = []
        for pred_box, true_box in zip(preds, targets):
            xA = max(true_box[0], pred_box[0])
            yA = max(true_box[1], pred_box[1])
            xB = min(true_box[2], pred_box[2])
            yB = min(true_box[3], pred_box[3])
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
            boxAArea = (true_box[2] - true_box[0] + 1) * (true_box[3] - true_box[1] + 1)
            boxBArea = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)
            iou = interArea / float(boxAArea + boxBArea - interArea)
            ious.append(iou)
        iou = np.mean(ious)
        return iou
