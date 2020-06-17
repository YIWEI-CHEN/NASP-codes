import torch

if __name__ == '__main__':
    import torchvision.models as models

    learning_rate = 0.2
    model = models.resnet50(num_classes=100)
    batch_size = 2048
    lr_multiplier = max(1.0, batch_size / 256)
    optimizer = torch.optim.SGD(
      model.parameters(),
      learning_rate,
      momentum=0.9,
      weight_decay=3e-4)
    steps = epochs = 50
    warmup = 4.0
    decay = 0.25

    def gradual_warmup_linear_scaling(step: int) -> float:
        epoch = step / float(epochs)

        # Gradual warmup
        warmup_ratio = min(warmup, epoch) / warmup
        multiplier = warmup_ratio * (lr_multiplier - 1.0) + 1.0

        if step < 17:
            return 1.0 * multiplier
        elif step < 33:
            return decay * multiplier
        elif step < 44:
            return decay ** 2 * multiplier
        return decay ** 3 * multiplier
        # if step < 15:
        #     return 1.0 * multiplier
        # elif step < 30:
        #     return decay * multiplier
        # elif step < 40:
        #     return decay ** 2 * multiplier
        # elif step < 45:
        #     return decay ** 3 * multiplier
        # return decay ** 4 * multiplier
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=gradual_warmup_linear_scaling)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs), eta_min=args.learning_rate_min)
    for i in range(epochs):
        lr = scheduler.get_lr()[0]
        # print(lr / learning_rate)
        print(lr)
        scheduler.step()
