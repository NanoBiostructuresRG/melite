# SPDX-License-Identifier: LGPL-3.0-or-later
"""XGBoost classifier preserving MELITE's original class identities."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted
from xgboost import XGBClassifier

__all__ = ["LabelIdentityXGBClassifier"]


class LabelIdentityXGBClassifier(XGBClassifier):
    """Encode labels for XGBoost and decode predictions transparently.

    Probability columns retain XGBoost's encoded order, which is exactly the
    deterministic order exposed through the original-label ``classes_``
    property.
    """

    @property
    def classes_(self):
        encoder = getattr(self, "label_encoder_", None)
        if encoder is not None:
            return encoder.classes_
        return super().classes_

    def fit(self, X, y, **kwargs):
        """Fit XGBoost using a training-local encoding of the supplied labels."""
        if hasattr(self, "label_encoder_"):
            del self.label_encoder_
        encoder = LabelEncoder().fit(y)
        if len(encoder.classes_) < 2:
            raise ValueError(
                "XGBoost classification requires at least two distinct classes."
            )
        encoded_y = encoder.transform(y)
        super().fit(X, encoded_y, **kwargs)
        self.label_encoder_ = encoder
        return self

    def predict(self, X, **kwargs):
        """Return predictions decoded to the original class identities."""
        check_is_fitted(self, "label_encoder_")
        encoded = super().predict(X, **kwargs)
        if kwargs.get("output_margin", False):
            return encoded
        return self.label_encoder_.inverse_transform(
            np.asarray(encoded, dtype=np.int64)
        )
