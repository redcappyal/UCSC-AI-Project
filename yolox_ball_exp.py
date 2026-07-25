"""YOLOX exp for the on-device ball detector. See ios/MODEL.md for the workflow.

    python -m yolox.tools.train -f yolox_ball_exp.py -d 1 -b 32 --fp16 -c yolox_tiny.pth

Not importable without YOLOX installed, which is deliberate: YOLOX is a training
dependency, not a runtime one, so it stays out of requirements.txt and out of the
test environment.

YOLOX (Apache-2.0) rather than Ultralytics YOLO (AGPL-3.0). The licence is the
reason, not the accuracy: AGPL is incompatible with shipping CrossCourt on the
App Store and, via section 13's network clause, with serving inference from the
Flask pipeline. Never initialise this from `yolo11n.pt`.

Every attribute below that differs from a YOLOX default carries the reason.
"""

from yolox.exp import Exp as MyExp

import os


class Exp(MyExp):
    def __init__(self):
        super().__init__()

        # --- model -------------------------------------------------------
        self.num_classes = 1                  # ball
        self.depth = 0.33
        self.width = 0.375                    # Tiny proportions
        # Nano sets depthwise=True. Kept dense because the Apple Neural Engine
        # runs dense convolutions more efficiently than depthwise stacks, and at
        # this size the parameter saving is not worth the latency risk.
        self.depthwise = False
        # SiLU, matching the COCO checkpoint this fine-tunes from. Activations
        # hold no parameters so ReLU would still load, but the pretrained
        # features were *learned* under SiLU and ReLU degrades them — and with
        # ~66 independent moments in the dataset there is no data to spare
        # recovering from that. SiLU is generally ANE-resident on recent chips.
        # Switch to "relu" and retrain ONLY if the Xcode performance report
        # actually shows SiLU falling back to CPU/GPU (ios/MODEL.md step 5).
        self.act = "silu"

        # --- data --------------------------------------------------------
        # Crops from prepare_ball_dataset.py; point at the generated folder.
        self.data_dir = os.environ.get("BALL_CROPS", "ball_crops")
        self.train_ann = "instances_train.json"
        self.val_ann = "instances_val.json"
        self.test_ann = "instances_test.json"
        # Image folders are train/, val/, test/ — not COCO's train2017/val2017/.
        # Read by the get_dataset/get_eval_dataset overrides below. A bare
        # `self.name` here would be dead config: nothing in YOLOX's Exp ever
        # reads it, so it silently would not redirect either loader.
        self.train_name = "train"
        self.val_name = "val"
        self.test_name = "test"

        # Crops are already cut at 416 px at native source resolution, so the
        # network input matches and nothing is resampled a second time.
        self.input_size = (416, 416)
        self.test_size = (416, 416)
        # No multiscale: apparent ball size is fixed by the rig's geometry, so
        # jittering scale only blurs the one scale that matters.
        self.multiscale_range = 0

        # --- augmentation ------------------------------------------------
        # Mosaic earns its place on small objects: four scenes per image means
        # more ball instances and more context boundaries per batch.
        self.mosaic_prob = 1.0
        self.mosaic_scale = (0.75, 1.25)      # narrow, per the multiscale note
        # Mixup blends two frames at ~50%. On a 10 px ball that halves an
        # already-marginal contrast against a white wall.
        self.enable_mixup = False
        self.mixup_prob = 0.0
        # The camera is bolted to the back wall glass fins with stabilisation
        # off, so the horizon never rotates. Rotation and shear would teach an
        # invariance that does not exist while resampling a tiny ball.
        self.degrees = 0.0
        self.shear = 0.0
        self.translate = 0.1
        # White balance is locked per court, so between-venue colour is the real
        # variation to model.
        self.hsv_prob = 1.0
        self.flip_prob = 0.5
        # NO motion-blur augmentation, ever. Real blur at 1/1000 s tops out near
        # aspect 2.4; the Roboflow v4 setting synthesised 100 px smears onto an
        # 8.7 px ball, erasing it while leaving the label attached, and cost
        # roughly two thirds of the model's recall.

        # --- schedule ----------------------------------------------------
        self.max_epoch = 300
        self.warmup_epochs = 5
        self.basic_lr_per_img = 0.01 / 64.0
        self.scheduler = "yoloxwarmcos"
        # Close mosaic for the tail so training finishes on real frames with
        # real backgrounds rather than composites.
        self.no_aug_epochs = 15
        self.ema = True
        self.weight_decay = 5e-4
        self.momentum = 0.9

        # --- eval --------------------------------------------------------
        self.eval_interval = 10
        # Single class and few candidates per crop, so NMS is cheap — and it
        # runs in Swift at inference, outside the Core ML graph, to keep the
        # graph ANE-resident.
        self.test_conf = 0.01
        self.nmsthre = 0.65

        self.print_interval = 50
        self.exp_name = "crosscourt-ball-416"

    # --- dataset wiring --------------------------------------------------
    # Both overrides exist for one reason: stock YOLOX hardcodes the COCO 2017
    # folder layout. Exp.get_dataset() passes no `name` at all, so COCODataset
    # falls back to its own "train2017" default, and Exp.get_eval_dataset()
    # hardcodes "val2017"/"test2017". Neither consults an exp attribute, so the
    # train/, val/, test/ folders this dataset ships are reachable only by
    # overriding. Annotation paths need no help — COCODataset already resolves
    # them as {data_dir}/annotations/{json_file}, matching the layout as built.

    def get_dataset(self, cache: bool = False, cache_type: str = "ram"):
        from yolox.data import COCODataset, TrainTransform

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.train_ann,
            name=self.train_name,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob,
            ),
            cache=cache,
            cache_type=cache_type,
        )

    def get_eval_dataset(self, **kwargs):
        from yolox.data import COCODataset, ValTransform

        testdev = kwargs.get("testdev", False)
        legacy = kwargs.get("legacy", False)

        return COCODataset(
            data_dir=self.data_dir,
            json_file=self.test_ann if testdev else self.val_ann,
            name=self.test_name if testdev else self.val_name,
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

    def get_model(self):
        # Overridden only to thread `depthwise` through. Stock Exp.get_model()
        # builds YOLOPAFPN/YOLOXHead without it, so self.depthwise would be dead
        # config — it agrees with their default today, but a future edit setting
        # it True would silently not apply. Otherwise identical to the base.
        import torch.nn as nn

        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(
                self.depth,
                self.width,
                in_channels=in_channels,
                act=self.act,
                depthwise=self.depthwise,
            )
            head = YOLOXHead(
                self.num_classes,
                self.width,
                in_channels=in_channels,
                act=self.act,
                depthwise=self.depthwise,
            )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model
