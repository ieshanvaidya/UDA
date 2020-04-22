import torch
from tqdm import tqdm


def get_tsa_threshold(schedule, t, T, K):
    """
    schedule: log, linear, exp
    t: training step
    T: total steps
    K: number of categories
    """
    step_ratio = torch.tensor(t / T)
    if schedule == 'log':
        alpha = 1 - torch.exp(-5 * step_ratio)
    elif schedule == 'linear':
        alpha = step_ratio
    elif schedule == 'exp':
        alpha = torch.exp(5 * (step_ratio - 1))
    else:
        raise ValueError('Invalid schedule')

    threshold = alpha * (1 - 1 / K) + 1 / K
    return threshold


def compute_loss(device, model, supervised_batch, unsupervised_batch, supervised_criterion, unsupervised_criterion,
                 training_step, args):
    model.train()

    supervised_input_ids, supervised_token_type_ids, supervised_attention_mask, supervised_labels = supervised_batch
    original_input_ids, original_token_type_ids, original_attention_mask, augmented_input_ids, augmented_token_type_ids, augmented_attention_mask = unsupervised_batch

    input_ids = torch.cat((supervised_input_ids, augmented_input_ids), dim=0).to(device)
    token_type_ids = torch.cat((supervised_token_type_ids, augmented_token_type_ids), dim=0).to(device)
    attention_mask = torch.cat((supervised_attention_mask, augmented_attention_mask), dim=0).to(device)
    supervised_labels = supervised_labels.to(device)

    original_input_ids = original_input_ids.to(device)
    original_token_type_ids = original_token_type_ids.to(device)
    original_attention_mask = original_attention_mask.to(device)

    logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]

    # Supervised Loss
    # ---------------
    num_supervised = supervised_labels.shape[0]
    supervised_loss = supervised_criterion(logits[:num_supervised], supervised_labels)

    threshold = get_tsa_threshold(args.schedule, training_step, args.total_steps, logits.shape[-1])

    # Probabilities: exp(-loss)
    larger_than_threshold = torch.exp(-supervised_loss) > threshold

    # Mask those below threshold
    supervised_mask = torch.ones_like(supervised_labels, dtype=torch.float32) * (
                1 - larger_than_threshold.type(torch.float32))

    # Recompute normalized loss
    supervised_loss = torch.sum(supervised_loss * supervised_mask, dim=-1) / torch.max(
        torch.sum(supervised_mask, dim=-1), torch.tensor(1.).to(device))

    # Unsupervised Loss
    # -----------------
    with torch.no_grad():
        original_logits = model(input_ids=original_input_ids, token_type_ids=original_token_type_ids,
                                attention_mask=original_attention_mask)[0]

        # KL divergence target
        original_probs = torch.softmax(original_logits, dim=-1)

        # Confidence based masking
        if args.uda_confidence_threshold != -1:
            unsupervised_mask = torch.max(original_probs, dim=-1)[0] > args.uda_confidence_threshold
            unsupervised_mask = unsupervised_mask.type(torch.float32)
        else:
            unsupervised_mask = torch.ones(len(logits) - num_supervised, dtype=torch.float32)

        unsupervised_mask = unsupervised_mask.to(device)

    uda_softmax_temp = args.uda_softmax_temperature if args.uda_softmax_temperature > 0 else 1
    augmented_log_probs = torch.log_softmax(logits[num_supervised:] / uda_softmax_temp, dim=-1)

    # Using SanghunYun's version (https://github.com/SanghunYun/UDA_pytorch)
    unsupervised_loss = torch.sum(unsupervised_criterion(augmented_log_probs, original_probs), dim=-1)
    unsupervised_loss = torch.sum(unsupervised_loss * unsupervised_mask, dim=-1) / torch.max(
        torch.sum(unsupervised_mask, dim=-1), torch.tensor(1.).to(device))

    final_loss = supervised_loss + args.uda_coefficient * unsupervised_loss

    return final_loss, supervised_loss, unsupervised_loss


def evaluate(device, model, dataloader):
    model.eval()

    correct = 0
    total = 0
    for batch in dataloader:
        input_ids, token_type_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        logits = model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]
        _, predictions = logits.max(1)

        correct += (predictions == labels).float().sum().item()
        total += input_ids.shape[0]

    return correct / total