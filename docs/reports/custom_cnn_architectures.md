# Custom CNN v1 versus v2

Both networks are trained **entirely from scratch** on the same 76-class merged
dataset, under the same deterministic split and the same 50-epoch budget. They
exist as a matched pair: v2 changes the architecture *and* the training recipe,
so the comparison isolates how much of the gap to a pretrained model is due to
pretraining and how much is due to methodology.

All shapes below are traced from the actual modules in
[`plant_classifier/models.py`](../../plant_classifier/models.py) at a
`224 x 224` RGB input.

---

## v1 — plain convolutional stack

Four convolutions, each followed by BatchNorm, ReLU, and (for the first three)
a max-pool. No skip connections, no attention. The spatial map is collapsed by
global average pooling straight into a linear classifier.

```mermaid
flowchart TD
    IN["Input<br/>3 x 224 x 224"] --> C1

    subgraph B1["Block 1"]
        C1["Conv 3x3, 32<br/>BatchNorm + ReLU"] --> P1["MaxPool 2<br/>32 x 112 x 112"]
    end

    subgraph B2["Block 2"]
        C2["Conv 3x3, 64<br/>BatchNorm + ReLU"] --> P2["MaxPool 2<br/>64 x 56 x 56"]
    end

    subgraph B3["Block 3"]
        C3["Conv 3x3, 128<br/>BatchNorm + ReLU"] --> P3["MaxPool 2<br/>128 x 28 x 28"]
    end

    subgraph B4["Block 4"]
        C4["Conv 3x3, 256<br/>BatchNorm + ReLU<br/>256 x 28 x 28"]
    end

    P1 --> C2
    P2 --> C3
    P3 --> C4
    C4 --> GAP["Global AvgPool<br/>256 x 1 x 1"]
    GAP --> DO["Dropout 0.3"]
    DO --> FC["Linear 256 -> 76"]
    FC --> OUT["76 class logits"]

    classDef conv fill:#dae8fc,stroke:#6c8ebf,color:#000
    classDef pool fill:#f5f5f5,stroke:#999,color:#000
    classDef head fill:#d5e8d4,stroke:#82b366,color:#000
    classDef io fill:#ffe6cc,stroke:#d79b00,color:#000
    class C1,C2,C3,C4 conv
    class P1,P2,P3,GAP pool
    class DO,FC head
    class IN,OUT io
```

**409K parameters · 4 convolutional layers · final feature map 28 x 28**

---

## v2 — residual stages with channel attention

The plain stack is replaced by a stem plus four residual stages, each holding
two `ResidualSEBlock`s. The classifier head is unchanged, so the head is *not*
a confounding variable in the comparison.

```mermaid
flowchart TD
    IN["Input<br/>3 x 224 x 224"] --> STEM["Stem<br/>Conv 3x3, 64 + BN + ReLU<br/>MaxPool 2 -> 64 x 112 x 112"]

    STEM --> S1["Stage 1<br/>2 x ResidualSE, stride 1<br/>64 x 112 x 112"]
    S1 --> S2["Stage 2<br/>2 x ResidualSE, stride 2<br/>128 x 56 x 56"]
    S2 --> S3["Stage 3<br/>2 x ResidualSE, stride 2<br/>256 x 28 x 28"]
    S3 --> S4["Stage 4<br/>2 x ResidualSE, stride 2<br/>512 x 14 x 14"]

    S4 --> GAP["Global AvgPool<br/>512 x 1 x 1"]
    GAP --> DO["Dropout 0.3"]
    DO --> FC["Linear 512 -> 76"]
    FC --> OUT["76 class logits"]

    classDef stage fill:#dae8fc,stroke:#6c8ebf,color:#000
    classDef pool fill:#f5f5f5,stroke:#999,color:#000
    classDef head fill:#d5e8d4,stroke:#82b366,color:#000
    classDef io fill:#ffe6cc,stroke:#d79b00,color:#000
    class STEM,S1,S2,S3,S4 stage
    class GAP pool
    class DO,FC head
    class IN,OUT io
```

**11.3M parameters · 20 convolutional layers · final feature map 14 x 14**

---

## Inside one ResidualSEBlock

This is where the two architectural ideas live. The **identity shortcut** gives
gradients a direct path backwards, so depth stops being an optimization
obstacle. The **SE module** rescales channels using global context, letting the
block emphasise whichever cues matter for the current image.

```mermaid
flowchart LR
    X["x<br/>in_ch"] --> CV1["Conv 3x3, stride s<br/>BatchNorm + ReLU"]
    CV1 --> CV2["Conv 3x3<br/>BatchNorm"]
    CV2 --> SE

    subgraph SE["SE module — channel attention"]
        direction TB
        SQ["Squeeze<br/>GlobalAvgPool -> 1 x 1 x C"]
        SQ --> EX["Excite<br/>Linear C -> C/16 -> ReLU<br/>Linear -> C -> Sigmoid"]
        EX --> SC["Scale<br/>multiply channel-wise"]
    end

    SE --> ADD(("+"))
    X -.->|"identity<br/>(1x1 conv if shape changes)"| ADD
    ADD --> RL["ReLU"] --> OUT["out<br/>out_ch"]

    classDef conv fill:#dae8fc,stroke:#6c8ebf,color:#000
    classDef se fill:#fff2cc,stroke:#d6b656,color:#000
    classDef op fill:#f8cecc,stroke:#b85450,color:#000
    class CV1,CV2 conv
    class SQ,EX,SC se
    class ADD,RL op
```

---

## What actually differs

```mermaid
flowchart TD
    subgraph SHARED["Held constant — not confounders"]
        H1["76-class merged dataset"]
        H2["Deterministic split, seed 42"]
        H3["224 x 224 input"]
        H4["50-epoch budget"]
        H5["GAP + Dropout 0.3 + Linear head"]
        H6["No pretrained weights"]
    end

    subgraph ARCH["Architecture changes"]
        A1["Residual shortcuts<br/>depth becomes trainable"]
        A2["SE channel attention<br/>per-image cue weighting"]
        A3["4 -> 20 conv layers<br/>409K -> 11.3M params"]
        A4["Final map 28x28 -> 14x14<br/>256 -> 512 channels"]
    end

    subgraph RECIPE["Training-recipe changes"]
        R1["Adam -> AdamW"]
        R2["Constant LR -> cosine warmup"]
        R3["Label smoothing 0.1"]
        R4["Cross-entropy -> class-balanced focal"]
        R5["MixUp 0.2 + CutMix 1.0"]
    end

    ARCH --> RES
    RECIPE --> RES
    RES["Result: macro F1 0.8730 -> 0.9637<br/>lab-to-field gap -17.5 -> -6.7 points"]

    classDef shared fill:#f5f5f5,stroke:#999,color:#000
    classDef arch fill:#dae8fc,stroke:#6c8ebf,color:#000
    classDef recipe fill:#fff2cc,stroke:#d6b656,color:#000
    classDef result fill:#d5e8d4,stroke:#82b366,color:#000
    class H1,H2,H3,H4,H5,H6 shared
    class A1,A2,A3,A4 arch
    class R1,R2,R3,R4,R5 recipe
    class RES result
```

---

## Measured outcome

| | v1 | v2 |
|---|---|---|
| Parameters | 409K | 11.3M |
| Conv layers | 4 | 20 |
| Final feature map | 256 x 28 x 28 | 512 x 14 x 14 |
| Top-1 accuracy | 0.9222 | **0.9655** |
| Macro F1 | 0.8730 | **0.9637** |
| Epochs to val macro F1 0.80 | 30 | **16** |
| Best val macro F1 | 0.8711 | **0.9627** |
| Leafsnap lab -> field gap | **-17.5** pts | **-6.7** pts |
| Grad-CAM attention drift | **-0.229** | **-0.137** |

Two readings matter for the thesis.

First, v2 reaches 0.9637 macro F1 **with no pretrained weights at all** —
within 1.1 points of fine-tuned ResNet50 (0.9747) and about 9 points above
frozen ImageNet features (0.8737). On a dataset of this size, methodology
recovers most of what transfer learning is usually credited with.

Second, v2 more than halves the lab-to-field gap (-17.5 to -6.7), beating
ResNet50 feature extraction (-7.1) on that axis. The Grad-CAM drift moves the
same way (-0.229 to -0.137). Aggressive augmentation perturbs exactly the
nuisance factors that separate lab from field capture, so it buys
cross-condition robustness that frozen pretrained features do not.

**Caveat.** The two changes — architecture and training recipe — were applied
together, so this comparison cannot attribute the improvement between them. An
ablation running the v2 architecture under the v1 recipe (and vice versa) is
needed to separate the two, and is listed as future work.
