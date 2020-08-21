from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, \
    get_constant_schedule_with_warmup


def get_schedule(args, optimizer, num_training_steps):
    num_warmup_steps = args.num_warmup_steps
    if 0 < num_warmup_steps and num_training_steps < 1:
        num_warmup_steps *= num_training_steps

    if args.schedule == 'linear':
        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                               num_training_steps=num_training_steps)
    elif args.schedule == 'cosine':
        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                               num_training_steps=num_training_steps, num_cycles=0.5)
    else:  # constant
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps)
