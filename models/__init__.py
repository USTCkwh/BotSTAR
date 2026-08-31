from .pretrain_model import PreModel


def build_model(args):
    input_dim = args.num_features
    num_heads_encoder = args.num_heads_encoder
    num_hidden = args.num_hidden
    attn_drop = args.attn_drop
    feat_drop = args.feat_drop
    replace_rate = args.replace_rate
    feat_mask_rate = args.feat_mask_rate
    leave_unchanged_rate = args.leave_unchanged_rate
    cluster_num = args.cluster_num
    interval = args.interval
    advanced_loss_rate = args.advanced_loss_rate
    instance_prototype_loss_rate = args.instance_prototype_loss_rate
    max_epoch = args.max_epoch
    use_align_loss = args.use_align_loss
    model = PreModel(
        dataset=args.dataset,
        input_dim=input_dim,
        hidden_dim=int(num_hidden),
        num_heads_encoder=num_heads_encoder,
        feat_drop=feat_drop,
        attn_drop=attn_drop,
        feat_mask_rate=feat_mask_rate,
        replace_rate=replace_rate,
        leave_unchanged_rate=leave_unchanged_rate,
        cluster_num=cluster_num,
        interval=interval,
        advanced_loss_rate=advanced_loss_rate,
        instance_prototype_loss_rate=instance_prototype_loss_rate,
        max_epoch=max_epoch,
        use_align_loss=use_align_loss,
        align_loss_rate=args.align_loss_rate,
    )
    return model
