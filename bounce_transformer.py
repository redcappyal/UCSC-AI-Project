"""PyTorch transformer classifier with a scikit-learn-compatible interface."""

from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


TEMPORAL_TOKEN_FEATURES = (
    "detected",
    "confidence",
    "x",
    "y",
    "width",
    "height",
)


def temporal_feature_columns(context):
    return [
        f"t{offset:+d}_{feature}"
        for offset in range(-int(context), int(context) + 1)
        for feature in TEMPORAL_TOKEN_FEATURES
    ]


def transformer_feature_columns(context, global_feature_columns):
    return temporal_feature_columns(context) + list(global_feature_columns)


class BounceTransformerNetwork(nn.Module):
    """Encode the local ball trajectory, then fuse engineered physics features."""

    def __init__(
        self,
        sequence_length,
        token_feature_count,
        global_feature_count,
        *,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        dropout,
    ):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead.")

        self.token_projection = nn.Sequential(
            nn.Linear(token_feature_count, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, sequence_length + 1, d_model)
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.temporal_norm = nn.LayerNorm(d_model)
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_count, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        nn.init.normal_(self.class_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def forward(self, sequence, global_features):
        tokens = self.token_projection(sequence)
        class_token = self.class_token.expand(sequence.shape[0], -1, -1)
        tokens = torch.cat((class_token, tokens), dim=1)
        tokens = tokens + self.position_embedding[:, : tokens.shape[1]]
        encoded = self.encoder(tokens)
        temporal_embedding = self.temporal_norm(encoded[:, 0])
        global_embedding = self.global_encoder(global_features)
        return self.classifier(
            torch.cat((temporal_embedding, global_embedding), dim=1)
        ).squeeze(1)


class BounceTransformerClassifier:
    """Trainable and joblib-safe wrapper exposing ``fit``/``predict_proba``."""

    def __init__(
        self,
        *,
        context,
        global_feature_columns,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.15,
        learning_rate=3e-4,
        weight_decay=1e-4,
        batch_size=128,
        epochs=40,
        validation_fraction=0.15,
        patience=7,
        random_seed=7,
        device="auto",
        verbose=True,
    ):
        self.context = int(context)
        self.global_feature_columns = list(global_feature_columns)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.validation_fraction = float(validation_fraction)
        self.patience = int(patience)
        self.random_seed = int(random_seed)
        self.device = str(device)
        self.verbose = bool(verbose)

    @property
    def sequence_length(self):
        return self.context * 2 + 1

    @property
    def sequence_feature_columns(self):
        return temporal_feature_columns(self.context)

    @property
    def feature_columns(self):
        return transformer_feature_columns(
            self.context,
            self.global_feature_columns,
        )

    def _as_dataframe(self, values):
        if isinstance(values, pd.DataFrame):
            return values
        columns = getattr(self, "feature_names_in_", self.feature_columns)
        return pd.DataFrame(values, columns=columns)

    def _arrays_from_features(self, values):
        frame = self._as_dataframe(values)
        missing = [column for column in self.feature_columns if column not in frame.columns]
        if missing:
            raise ValueError(
                "Transformer input is missing feature column(s): "
                + ", ".join(missing)
            )

        sequence_channels = []
        for offset in range(-self.context, self.context + 1):
            columns = [
                f"t{offset:+d}_{feature}"
                for feature in TEMPORAL_TOKEN_FEATURES
            ]
            sequence_channels.append(
                frame[columns].to_numpy(dtype=np.float32, copy=True)
            )
        sequence = np.stack(sequence_channels, axis=1)
        global_features = frame[self.global_feature_columns].to_numpy(
            dtype=np.float32,
            copy=True,
        )
        return (
            np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0),
            np.nan_to_num(global_features, nan=0.0, posinf=0.0, neginf=0.0),
        )

    def _resolved_device(self):
        if self.device != "auto":
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _normalize(self, sequence, global_features, *, fit):
        if fit:
            self.sequence_mean_ = sequence.mean(axis=(0, 1), keepdims=True)
            self.sequence_scale_ = sequence.std(axis=(0, 1), keepdims=True)
            self.sequence_scale_[self.sequence_scale_ < 1e-6] = 1.0
            self.global_mean_ = global_features.mean(axis=0, keepdims=True)
            self.global_scale_ = global_features.std(axis=0, keepdims=True)
            self.global_scale_[self.global_scale_ < 1e-6] = 1.0

        return (
            (sequence - self.sequence_mean_) / self.sequence_scale_,
            (global_features - self.global_mean_) / self.global_scale_,
        )

    def _build_network(self):
        return BounceTransformerNetwork(
            self.sequence_length,
            len(TEMPORAL_TOKEN_FEATURES),
            len(self.global_feature_columns),
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        )

    @staticmethod
    def _weighted_loss(logits, labels, weights):
        losses = nn.functional.binary_cross_entropy_with_logits(
            logits,
            labels,
            reduction="none",
        )
        return (losses * weights).sum() / weights.sum().clamp_min(1e-8)

    def _validation_loss(self, loader, device):
        self.network_.eval()
        total_loss = 0.0
        total_weight = 0.0
        with torch.no_grad():
            for sequence, global_features, labels, weights in loader:
                sequence = sequence.to(device)
                global_features = global_features.to(device)
                labels = labels.to(device)
                weights = weights.to(device)
                logits = self.network_(sequence, global_features)
                loss = self._weighted_loss(logits, labels, weights)
                batch_weight = float(weights.sum().item())
                total_loss += float(loss.item()) * batch_weight
                total_weight += batch_weight
        return total_loss / max(total_weight, 1e-8)

    def fit(self, values, labels, sample_weight=None):
        sequence, global_features = self._arrays_from_features(values)
        labels = np.asarray(labels, dtype=np.float32)
        if labels.ndim != 1 or len(labels) != len(sequence):
            raise ValueError("labels must contain one value per feature row.")
        if len(np.unique(labels)) < 2:
            raise ValueError("Transformer training requires both classes.")

        if sample_weight is None:
            sample_weight = np.ones(len(labels), dtype=np.float32)
        sample_weight = np.asarray(sample_weight, dtype=np.float32)
        if sample_weight.shape != labels.shape:
            raise ValueError("sample_weight must contain one value per label.")

        sequence, global_features = self._normalize(
            sequence,
            global_features,
            fit=True,
        )
        self.feature_names_in_ = np.asarray(self.feature_columns, dtype=object)
        self.classes_ = np.asarray([0, 1], dtype=np.int64)

        indices = np.arange(len(labels))
        use_validation = (
            self.validation_fraction > 0
            and len(labels) >= 20
            and min(np.bincount(labels.astype(int))) >= 2
        )
        if use_validation:
            train_indices, validation_indices = train_test_split(
                indices,
                test_size=self.validation_fraction,
                random_state=self.random_seed,
                stratify=labels,
            )
        else:
            train_indices = indices
            validation_indices = indices

        def dataset_for(dataset_indices):
            return TensorDataset(
                torch.from_numpy(sequence[dataset_indices]).float(),
                torch.from_numpy(global_features[dataset_indices]).float(),
                torch.from_numpy(labels[dataset_indices]).float(),
                torch.from_numpy(sample_weight[dataset_indices]).float(),
            )

        generator = torch.Generator().manual_seed(self.random_seed)
        train_loader = DataLoader(
            dataset_for(train_indices),
            batch_size=max(1, self.batch_size),
            shuffle=True,
            generator=generator,
        )
        validation_loader = DataLoader(
            dataset_for(validation_indices),
            batch_size=max(1, self.batch_size),
            shuffle=False,
        )

        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        device = self._resolved_device()
        self.network_ = self._build_network().to(device)
        optimizer = torch.optim.AdamW(
            self.network_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        best_state = None
        best_validation_loss = float("inf")
        epochs_without_improvement = 0
        self.training_history_ = []
        for epoch in range(1, max(1, self.epochs) + 1):
            self.network_.train()
            total_loss = 0.0
            total_weight = 0.0
            for batch_sequence, batch_global, batch_labels, batch_weights in train_loader:
                batch_sequence = batch_sequence.to(device)
                batch_global = batch_global.to(device)
                batch_labels = batch_labels.to(device)
                batch_weights = batch_weights.to(device)

                optimizer.zero_grad(set_to_none=True)
                logits = self.network_(batch_sequence, batch_global)
                loss = self._weighted_loss(logits, batch_labels, batch_weights)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network_.parameters(), max_norm=1.0)
                optimizer.step()

                batch_weight = float(batch_weights.sum().item())
                total_loss += float(loss.item()) * batch_weight
                total_weight += batch_weight

            train_loss = total_loss / max(total_weight, 1e-8)
            validation_loss = self._validation_loss(validation_loader, device)
            self.training_history_.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                }
            )
            if self.verbose:
                print(
                    f"Transformer epoch {epoch:03d}/{self.epochs}: "
                    f"train_loss={train_loss:.5f}, "
                    f"validation_loss={validation_loss:.5f}",
                    flush=True,
                )

            if validation_loss < best_validation_loss - 1e-6:
                best_validation_loss = validation_loss
                best_state = deepcopy(self.network_.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if self.patience > 0 and epochs_without_improvement >= self.patience:
                    if self.verbose:
                        print(
                            f"Transformer early stopping after epoch {epoch}.",
                            flush=True,
                        )
                    break

        if best_state is not None:
            self.network_.load_state_dict(best_state)
        self.network_.to("cpu")
        self.network_.eval()
        self.best_validation_loss_ = best_validation_loss
        self.n_features_in_ = len(self.feature_columns)
        return self

    def predict_proba(self, values):
        if not hasattr(self, "network_"):
            raise RuntimeError("BounceTransformerClassifier has not been fitted.")
        sequence, global_features = self._arrays_from_features(values)
        sequence, global_features = self._normalize(
            sequence,
            global_features,
            fit=False,
        )
        dataset = TensorDataset(
            torch.from_numpy(sequence).float(),
            torch.from_numpy(global_features).float(),
        )
        loader = DataLoader(
            dataset,
            batch_size=max(1, self.batch_size),
            shuffle=False,
        )

        probabilities = []
        self.network_.eval()
        with torch.no_grad():
            for batch_sequence, batch_global in loader:
                logits = self.network_(batch_sequence, batch_global)
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
        positive = (
            np.concatenate(probabilities)
            if probabilities
            else np.empty(0, dtype=np.float32)
        )
        return np.column_stack((1.0 - positive, positive))

    def predict(self, values):
        return (self.predict_proba(values)[:, 1] >= 0.5).astype(np.int64)
