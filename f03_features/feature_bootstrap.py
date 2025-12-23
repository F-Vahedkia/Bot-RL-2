# f03_features/feature_bootstrap

# این فایل بیرون از indicators است تا لوپ ایمپورت نشود
from f03_features.feature_registry import REGISTRY  # رجیستری اندیکاتورهای شما (تست‌شده)
from f03_features.price_action.registry_adapter import register_price_action_to_indicators_registry

# Attach PA builders to the main registry (no overwrite)
register_price_action_to_indicators_registry(REGISTRY)

# Use REGISTRY after importing this module, e.g.:
#   from f03_features.feature_bootstrap import REGISTRY
# Example:
#   REGISTRY.get("pa_market_structure")
#
# Purpose: attach Price Action builders to the tested indicators REGISTRY
# without modifying indicators package and without import loops.
#
# Load this module (feature_bootstrap) before using REGISTRY in your app.
# Do NOT import this module inside f03_features/indicators/*.
# Keep this file thin (no extra logic); it’s only for registration wiring.
