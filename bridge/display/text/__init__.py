from bridge.display.text.classification import TextClassificationPanelEngine

# Back-compat alias. The old generic name is preserved so existing notebooks,
# tests, and provider defaults keep working until feature/notebooks-rewrite
# updates everything to the new names.
Panel = TextClassificationPanelEngine

__all__ = ["TextClassificationPanelEngine", "Panel"]
