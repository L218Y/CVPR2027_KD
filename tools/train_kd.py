"""Train an RGB-only student with recoverability-aware privileged distillation.

The multimodal teacher is frozen.  The student only ever sees ``img``; the
auxiliary modalities are used during training to estimate which parts of the
teacher's evidence RGB can actually recover, and to decide what to distil.

Usage (single GPU):
    python tools/train_kd.py --cfg configs/kd_mfnet.yaml

Usage (DDP, 4 GPUs):
    python -m torch.distributed.launch --nproc_per_node=4 --use_env \
        tools/train_kd.py --cfg configs/kd_mfnet.yaml
"""

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml
from tabulate import tabulate
from torch import nn
from torch.backends import cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from semseg.augmentations_mm import get_train_augmentation, get_val_augmentation
from semseg.datasets import *
from semseg.datasets.stats import train_label_counts
from semseg.kd import (
    FeatureProbe,
    RecoverabilityHead,
    RecoverabilityKD,
    PrivilegedInjector,
    TaskAwareKD,
    UniformKD,
    class_balance_weights,
)
from semseg.losses import get_loss
from semseg.models import *
from semseg.optimizers import get_optimizer
from semseg.schedulers import get_scheduler
from semseg.utils.utils import cleanup_ddp, fix_seeds, get_logger, print_iou, setup_cudnn, setup_ddp
from val_mm import evaluate

try:
    from torch.amp import GradScaler, autocast

    def _scaler(enabled):
        return GradScaler("cuda", enabled=enabled)

    def _autocast(enabled):
        return autocast("cuda", enabled=enabled)

except Exception:  # pragma: no cover - older torch
    from torch.cuda.amp import GradScaler, autocast

    def _scaler(enabled):
        return GradScaler(enabled=enabled)

    def _autocast(enabled):
        return autocast(enabled=enabled)


def load_state(path, model, strict=True):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break

    def norm(k):
        k = k.replace("module.", "", 1)
        return k.replace(".module.", ".")

    target = model.state_dict()
    tgt = {norm(k): k for k in target}
    mapped, used = {}, set()
    for k, v in ckpt.items():
        nk = norm(k)
        if nk in tgt and target[tgt[nk]].shape == v.shape:
            mapped[tgt[nk]] = v
            used.add(nk)
    missing = [k for k in tgt if k not in used]
    unused = [k for k in ckpt if norm(k) not in tgt or target[tgt[norm(k)]].shape != ckpt[k].shape]
    if missing and strict:
        raise RuntimeError(f"checkpoint '{path}' is missing {len(missing)} tensors, e.g. {missing[:4]}")
    if unused and strict:
        raise RuntimeError(
            f"checkpoint '{path}' has {len(unused)} tensors the model cannot take, e.g. {unused[:4]}; "
            "this usually means the checkpoint was built with different modalities"
        )
    model.load_state_dict(mapped, strict=False)
    return (f"loaded {len(mapped)}/{len(target)} tensors "
            f"(missing {len(missing)}, unexpected {len(unused)})")


def build_model(cfg, n_classes, modals, backbone=None, pretrained=None):
    backbone = backbone or cfg["MODEL"]["BACKBONE"]
    model = eval(cfg["MODEL"]["NAME"])(backbone, n_classes, modals)
    if pretrained:
        model.init_pretrained(pretrained)
    return model


def main(cfg, gpu, save_dir, args):
    start = time.time()
    device = torch.device(cfg["DEVICE"] if torch.cuda.is_available() else "cpu")
    train_cfg, eval_cfg = cfg["TRAIN"], cfg["EVAL"]
    dataset_cfg, model_cfg = cfg["DATASET"], cfg["MODEL"]
    loss_cfg, optim_cfg, sched_cfg = cfg["LOSS"], cfg["OPTIMIZER"], cfg["SCHEDULER"]
    kd_cfg = cfg.get("KD", {})

    if train_cfg.get("DETERMINISTIC", False):
        cudnn.benchmark = False
        cudnn.deterministic = True

    epochs = int(args.epochs or train_cfg["EPOCHS"])
    lr = optim_cfg["LR"]
    resume_path = cfg["MODEL"].get("RESUME", "")
    distributed = dist.is_available() and dist.is_initialized()
    world = dist.get_world_size() if distributed else 1

    teacher_modals = kd_cfg.get("TEACHER_MODALS", dataset_cfg["MODALS"])
    use_kd = bool(kd_cfg.get("ENABLE", True))
    # With KD off there is no teacher: the student must follow the dataset's own
    # modality list. Using teacher_modals here silently built a two-stream student
    # for CE baselines whose config still carried KD.TEACHER_MODALS.
    student_modals = ["img"] if use_kd else list(dataset_cfg["MODALS"])
    assert teacher_modals[0] == "img", "the first modality must be RGB"

    traintransform = get_train_augmentation(train_cfg["IMAGE_SIZE"], seg_fill=dataset_cfg["IGNORE_LABEL"])
    valtransform = get_val_augmentation(eval_cfg["IMAGE_SIZE"])

    dataset_cls = eval(dataset_cfg["NAME"])
    trainset = dataset_cls(dataset_cfg["ROOT"], "train", traintransform, teacher_modals)
    valset = dataset_cls(dataset_cfg["ROOT"], "val", valtransform, student_modals)
    class_names = trainset.CLASSES
    n_classes = trainset.n_classes

    student = build_model(cfg, n_classes, student_modals, pretrained=model_cfg.get("PRETRAINED"))
    teacher = None
    if use_kd:
        teacher = build_model(
            cfg,
            n_classes,
            teacher_modals,
            backbone=kd_cfg.get("TEACHER_BACKBONE", model_cfg["BACKBONE"]),
            pretrained=None,
        )
        teacher_path = kd_cfg.get("TEACHER_PATH", "")
        if teacher_path and os.path.isfile(teacher_path):
            msg = load_state(teacher_path, teacher, strict=True)
            logger.info(f"teacher loaded: {msg}")
        elif kd_cfg.get("ALLOW_RANDOM_TEACHER", False):
            logger.warning(f"teacher checkpoint not found at '{teacher_path}'; using random weights (smoke test only)")
        else:
            raise FileNotFoundError(f"teacher checkpoint required but missing: '{teacher_path}'")
        teacher.to(device).eval()
        for p in teacher.parameters():
            p.requires_grad_(False)

    student.to(device)
    if os.path.isfile(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        student.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"student resumed from {resume_path}")

    s_dim = student.backbone.channels[-1]
    probe, rec_head, kd_loss = None, None, None
    kd_mode = kd_cfg.get("MODE", "recoverability") if use_kd else "none"
    balance_cfg = loss_cfg.get("CLASS_BALANCE") or {}
    balance_on = bool(balance_cfg.get("ENABLE", False))
    kd_gate = str(kd_cfg.get("GATE", "none"))
    counts, class_weights = None, None
    if balance_on or (kd_mode == "task_aware" and "class_balance" in kd_gate):
        counts = train_label_counts(trainset, cache_dir=Path("results/protocol"))
        class_weights = class_balance_weights(
            counts, beta=float(balance_cfg.get("BETA", 0.5)), cap=float(balance_cfg.get("CAP", 5.0))
        ).to(device)
        logger.info(
            "train label weights: "
            + json.dumps({c: round(float(w), 4) for c, w in zip(class_names, class_weights.tolist())})
        )
    injector, injector_drop = None, 0.0
    inject_cfg = kd_cfg.get("PRIV_INJECT") or {}
    if use_kd and bool(inject_cfg.get("ENABLE", False)):
        # Training-time side channel only: absent at inference, so the deployed
        # student stays RGB-only. Gate starts at zero so the run begins as the
        # RGB baseline and adopts the side channel only if it pays for itself.
        injector = PrivilegedInjector(
            teacher.backbone.channels[-1],
            student.backbone.channels[-1],
            gate_init=float(inject_cfg.get("GATE_INIT", 0.0)),
        ).to(device)
        injector_drop = float(inject_cfg.get("DROP_PROB", 0.5))
        logger.info(f"privileged injection enabled (drop_prob={injector_drop})")

    if use_kd:
        t_dim = teacher.backbone.channels[-1]
        # The probe is only meaningful for the recoverability family; task-level
        # gating needs no learned feature predictor, so it is skipped entirely.
        if kd_mode in ("recoverability", "conservative_only"):
            probe = FeatureProbe(s_dim, t_dim, hidden_dim=kd_cfg.get("PROBE_HIDDEN", 128)).to(device)
        if kd_cfg.get("DEPLOY_HEAD", False):
            rec_head = RecoverabilityHead(s_dim).to(device)
        if kd_mode == "uniform":
            kd_loss = UniformKD(tau=kd_cfg.get("TAU", 2.0)).to(device)
        elif kd_mode == "task_aware":
            kd_loss = TaskAwareKD(
                tau=kd_cfg.get("TAU", 2.0),
                gate=kd_gate,
                class_weights=class_weights if class_weights is not None else torch.ones(n_classes),
                ignore_index=trainset.ignore_label,
                foreground_only=bool(kd_cfg.get("FOREGROUND_ONLY", False)),
            ).to(device)
        else:
            kd_loss = RecoverabilityKD(
                tau=kd_cfg.get("TAU", 2.0),
                eps_max=kd_cfg.get("EPS_MAX", 0.5),
                mode=kd_mode,
                gate=kd_gate,
                rare_protect=float(kd_cfg.get("RARE_PROTECT", 0.0)),
            ).to(device)

    iters_per_epoch = len(trainset) // train_cfg["BATCH_SIZE"] // world
    ce_loss = get_loss(loss_cfg["NAME"], trainset.ignore_label, class_weights if balance_on else None)
    optimizer = get_optimizer(student, optim_cfg["NAME"], lr, optim_cfg["WEIGHT_DECAY"])
    if injector is not None:
        optimizer.add_param_group({"params": list(injector.parameters())})
    probe_params = (list(probe.parameters()) if probe else []) + (list(rec_head.parameters()) if rec_head else [])
    probe_optimizer = None
    if probe_params:
        probe_optimizer = torch.optim.AdamW(
            probe_params, lr=kd_cfg.get("PROBE_LR", 1e-3), weight_decay=0.01
        )
    scheduler = get_scheduler(
        sched_cfg["NAME"], optimizer, int((epochs + 1) * iters_per_epoch),
        sched_cfg["POWER"], iters_per_epoch * sched_cfg["WARMUP"], sched_cfg["WARMUP_RATIO"],
    )

    # Data order and augmentation randomness come from a generator that is not
    # shared with module construction: building a teacher or a probe consumes
    # draws from the global RNG, so arms would otherwise see different batches
    # despite the same --seed.
    data_generator = torch.Generator()
    data_generator.manual_seed(args.seed)

    if distributed:
        sampler = DistributedSampler(trainset, world, dist.get_rank(), shuffle=True)
        student = DDP(student, device_ids=[gpu], output_device=gpu, find_unused_parameters=True)
        if probe is not None:
            probe = DDP(probe, device_ids=[gpu], output_device=gpu)
    else:
        sampler = RandomSampler(trainset, generator=data_generator)

    trainloader = DataLoader(
        trainset, batch_size=train_cfg["BATCH_SIZE"], num_workers=args.workers,
        drop_last=True, pin_memory=True, sampler=sampler, generator=data_generator,
    )
    valloader = DataLoader(
        valset, batch_size=eval_cfg["BATCH_SIZE"], num_workers=args.workers,
        pin_memory=True, sampler=None,
    )

    scaler = _scaler(train_cfg.get("AMP", False))
    writer = None
    if (distributed and dist.get_rank() == 0) or not distributed:
        writer = SummaryWriter(str(save_dir))
        logger.info(json.dumps(cfg, indent=2))

    best_miou, best_epoch, best_acc, best_macc = 0.0, 0, 0.0, 0.0
    eval_history = []
    warmup = int(kd_cfg.get("WARMUP_EPOCHS", 5))
    lambda_kd = float(kd_cfg.get("LAMBDA_KD", 1.0))
    lambda_feat = float(kd_cfg.get("LAMBDA_FEAT", 0.0))
    lambda_probe = float(kd_cfg.get("LAMBDA_PROBE", 0.1))
    ignore_index = trainset.ignore_label

    # Re-seed the global RNG so dropout masks (SegFormerHead uses Dropout2d) are
    # identical across arms; everything drawn before this point differs because
    # some arms build a teacher or a probe and others do not.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    stats = {}
    for epoch in range(epochs):
        student.train()
        if probe is not None:
            probe.train()
        if rec_head:
            rec_head.train()
        if distributed:
            sampler.set_epoch(epoch)

        running = {"loss": 0.0, "kd": 0.0, "r": 0.0, "probe": 0.0, "sel": 0.0, "inj": 0.0}
        # task-level gates need no warm-up: they depend on the label and the
        # teacher posterior, not on a probe that has to learn first.
        use_rec = True if kd_mode == "task_aware" else epoch >= warmup
        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch, desc=f"Epoch {epoch + 1}/{epochs}")
        it = -1
        for it, (sample, lbl) in pbar:
            sample = [x.to(device, non_blocking=True) for x in sample]
            lbl = lbl.to(device, non_blocking=True)
            rgb = sample[:1]
            optimizer.zero_grad(set_to_none=True)
            if probe_optimizer is not None:
                probe_optimizer.zero_grad(set_to_none=True)

            with _autocast(train_cfg.get("AMP", False)):
                model_ref = student.module if distributed else student
                s_feats = model_ref.backbone(sample if not use_kd else rgb)
                t_feats, t_logits, injected = None, None, False
                if use_kd:
                    with torch.no_grad():
                        t_feats = teacher.backbone(sample)
                        t_logits = teacher.decode_head(t_feats)
                        t_logits = F.interpolate(t_logits, size=lbl.shape[-2:], mode="bilinear", align_corners=False)
                    if injector is not None and torch.rand((), device=device) >= injector_drop:
                        injector_module = injector.module if distributed else injector
                        s_feats[-1] = s_feats[-1] + injector_module(t_feats[-1].detach(), s_feats[-1])
                        injected = True
                logits = model_ref.decode_head(s_feats)
                logits = F.interpolate(logits, size=lbl.shape[-2:], mode="bilinear", align_corners=False)
                loss = ce_loss(logits, lbl)
                kd_out = {"kd": torch.zeros((), device=device), "r_mean": torch.zeros((), device=device)}
                probe_mse = torch.zeros((), device=device)
                r_small, r_map = None, None
                if use_kd:
                    probe_module = (probe.module if distributed else probe) if probe is not None else None
                    if probe_module is not None:
                        r_small, probe_mse = probe_module(s_feats[-1].detach(), t_feats[-1])
                        r_map = F.interpolate(
                            r_small.unsqueeze(1), size=lbl.shape[-2:], mode="bilinear", align_corners=False
                        )
                    kd_out = kd_loss(
                        logits, t_logits, r=r_map if use_rec else None, labels=lbl, ignore_index=ignore_index,
                        student_feat=s_feats[-1] if lambda_feat > 0 else None,
                        teacher_feat=t_feats[-1] if lambda_feat > 0 else None,
                        proj=probe_module.proj if (lambda_feat > 0 and probe_module is not None) else None,
                        lambda_feat=lambda_feat,
                    )
                    loss = loss + lambda_kd * kd_out["total"]
                    if probe_module is not None:
                        loss = loss + lambda_probe * probe_mse
                    if rec_head is not None and use_rec and r_small is not None:
                        r_pred = rec_head(s_feats[-1])
                        loss = loss + float(kd_cfg.get("LAMBDA_HEAD", 0.1)) * F.mse_loss(
                            r_pred, r_small.detach().unsqueeze(1)
                        )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            if probe_optimizer is not None:
                scaler.step(probe_optimizer)
            scaler.update()
            scheduler.step()

            running["loss"] += loss.item()
            running["kd"] += float(kd_out["kd"].detach())
            running["r"] += float(kd_out["r_mean"]) if "r_mean" in kd_out else 0.0
            running["sel"] += float(kd_out.get("select_frac", torch.zeros(())))
            running["probe"] += float(probe_mse.detach())
            running["inj"] += 1.0 if injected else 0.0
            if args.max_iters and (it + 1) >= args.max_iters:
                break
            if it % 10 == 0:
                pbar.set_description(
                    f"ep {epoch + 1}/{epochs} loss {running['loss'] / (it + 1):.4f} "
                    f"kd {running['kd'] / (it + 1):.4f} r {running['r'] / (it + 1):.3f}"
                )

        n = max(1, it + 1)
        stats = {k: v / n for k, v in running.items()}
        if it < 0:
            logger.warning("trainloader produced no batches; skipping epoch statistics")
        if writer:
            for k, v in stats.items():
                writer.add_scalar(f"train/{k}", v, epoch)

        should_eval = not args.skip_eval and (
            ((epoch + 1) % train_cfg.get("EVAL_INTERVAL", 1) == 0 and
             (epoch + 1) > train_cfg.get("EVAL_START", 0)) or (epoch + 1) == epochs)
        if should_eval and ((distributed and dist.get_rank() == 0) or not distributed):
            acc, macc, _, _, ious, miou = evaluate(student, valloader, device)
            if writer:
                writer.add_scalar("val/mIoU", miou, epoch)
            logger.info(f"epoch {epoch + 1} mIoU {miou:.3f} mAcc {macc:.3f} best {best_miou:.3f}")
            eval_history.append((epoch + 1, float(miou)))
            sd = student.module.state_dict() if distributed else student.state_dict()
            torch.save({"model_state_dict": sd, "epoch": epoch + 1, "miou": miou}, save_dir / "last.pth")
            if probe is not None:
                pmod = probe.module if distributed else probe
                torch.save(pmod.state_dict(), save_dir / "probe.pth")
            if miou > best_miou:
                best_miou, best_epoch, best_acc, best_macc = miou, epoch + 1, acc, macc
                torch.save(sd, save_dir / "best.pth")
                logger.info(print_iou(epoch, ious, miou, acc, macc, class_names))

    if writer:
        writer.close()

    if (distributed and dist.get_rank() == 0) or not distributed:
        elapsed = time.gmtime(time.time() - start)

        def sha256(path):
            p = Path(path)
            return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None

        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
            ).stdout.strip()
        except Exception:
            commit = None
        summary = {
            "experiment": save_dir.name,
            "dataset": dataset_cfg["NAME"],
            "backbone": model_cfg["BACKBONE"],
            "teacher_backbone": kd_cfg.get("TEACHER_BACKBONE", model_cfg["BACKBONE"]),
            "teacher_path": kd_cfg.get("TEACHER_PATH", "") if use_kd else "",
            "teacher_sha256": sha256(kd_cfg.get("TEACHER_PATH", "")) if use_kd else None,
            "modals": teacher_modals,
            "student_modals": student_modals,
            "kd_mode": kd_mode,
            "gate": kd_gate if use_kd else "none",
            "rare_protect": float(kd_cfg.get("RARE_PROTECT", 0.0)),
            "tau": kd_cfg.get("TAU", 2.0),
            "eps_max": kd_cfg.get("EPS_MAX", 0.5),
            "lambda_kd": lambda_kd,
            "lambda_feat": lambda_feat,
            "priv_inject": {
                "enabled": injector is not None,
                "drop_prob": injector_drop,
                "gate": None if injector is None else float(injector.gate.detach()),
            },
            "class_balance": {
                "enabled": balance_on,
                "beta": float(balance_cfg.get("BETA", 0.5)),
                "cap": float(balance_cfg.get("CAP", 5.0)),
                "weights": None if class_weights is None else {
                    c: round(float(w), 4) for c, w in zip(class_names, class_weights.tolist())
                },
                "counts": None if counts is None else {
                    c: int(n) for c, n in zip(class_names, counts.tolist())
                },
            },
            "seed": args.seed,
            "epochs": epochs,
            "batch_size": train_cfg["BATCH_SIZE"],
            "world_size": world,
            "best_miou": best_miou,
            "last10_miou": float(sum(m for _, m in eval_history[-10:]) / max(len(eval_history[-10:]), 1)),
            "eval_history": eval_history,
            "mAcc": best_macc,
            "acc_per_class": [round(float(x), 2) for x in best_acc]
            if isinstance(best_acc, (list, tuple))
            else [],
            "best_epoch": best_epoch,
            "train_time": time.strftime("%H:%M:%S", elapsed),
            "final_stats": stats,
            "provenance": {
                "config": args.cfg,
                "config_sha256": sha256(args.cfg),
                "code_commit": commit,
                "torch": torch.__version__,
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "deterministic": bool(train_cfg.get("DETERMINISTIC", False)),
                "train_split_sha256": sha256(Path(dataset_cfg["ROOT"]) / "train.txt"),
                "eval_split_file_sha256": sha256(Path(dataset_cfg["ROOT"]) / "test.txt"),
                "n_train": len(trainset),
                "n_eval": len(valset),
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start)),
            },
        }
        (save_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
        results_dir = Path("results") / save_dir.name
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
        logger.info(tabulate([["Best mIoU", f"{best_miou:.2f}"], ["Time", summary["train_time"]]]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/kd_mfnet.yaml")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=0, help="override config epochs (smoke tests)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tag", type=str, default="", help="suffix for the experiment name")
    parser.add_argument("--max-iters", type=int, default=0, help="stop an epoch early (smoke tests)")
    parser.add_argument("--skip-eval", action="store_true", help="disable validation (smoke tests)")
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds(args.seed)
    setup_cudnn()
    gpu = setup_ddp()
    modals = "".join([m[0] for m in cfg["DATASET"]["MODALS"]])
    kd_cfg = cfg.get("KD", {})
    if kd_cfg.get("ENABLE", True):
        stage = kd_cfg.get("MODE", "kd")
    else:
        stage = "rgb" if len(cfg["DATASET"]["MODALS"]) == 1 else "teacher"
    exp_name = "_".join([
        cfg["DATASET"]["NAME"], cfg["MODEL"]["BACKBONE"], modals, stage, f"s{args.seed}",
    ])
    if args.tag:
        exp_name += f"_{args.tag}"
    save_dir = Path(cfg["SAVE_DIR"], exp_name)
    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger(save_dir / "train.log")
    main(cfg, gpu, save_dir, args)
    cleanup_ddp()
