"""
Tests for core/registry.py: Registry class, component registration,
search, aliases, create_from_config, and decorator functions.
"""

import pytest

from alignment.core.registry import ComponentInfo, Registry, create_component, create_from_config, list_all_components

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyMetric:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def compute(self):
        return self.alpha


class _DummyModel:
    pass


# =========================================================================
# Registry core
# =========================================================================


class TestRegistry:
    def test_register_and_get(self):
        reg = Registry("test")
        reg.register("foo", _DummyMetric)
        assert reg.get("foo") is _DummyMetric

    def test_register_as_decorator(self):
        reg = Registry("test")

        @reg.register("bar", category="info")
        class Bar:
            pass

        assert reg.get("bar") is Bar

    def test_get_missing_raises(self):
        reg = Registry("test")
        with pytest.raises(KeyError, match="not found"):
            reg.get("nonexistent")

    def test_list_all(self):
        reg = Registry("test")
        reg.register("a", _DummyMetric)
        reg.register("b", _DummyModel)
        assert set(reg.list()) == {"a", "b"}

    def test_list_by_category(self):
        reg = Registry("test")
        reg.register("a", _DummyMetric, category="info")
        reg.register("b", _DummyModel, category="arch")
        assert reg.list(category="info") == ["a"]
        assert reg.list(category="arch") == ["b"]

    def test_list_categories(self):
        reg = Registry("test")
        reg.register("a", _DummyMetric, category="info")
        reg.register("b", _DummyModel, category="arch")
        cats = reg.list_categories()
        assert set(cats) == {"info", "arch"}

    def test_aliases(self):
        reg = Registry("test")
        reg.register("full_name", _DummyMetric, aliases=["short", "alias2"])
        assert reg.get("short") is _DummyMetric
        assert reg.get("alias2") is _DummyMetric
        assert reg.resolve_name("short") == "full_name"

    def test_contains(self):
        reg = Registry("test")
        reg.register("x", _DummyMetric, aliases=["y"])
        assert "x" in reg
        assert "y" in reg
        assert "z" not in reg

    def test_len(self):
        reg = Registry("test")
        assert len(reg) == 0
        reg.register("a", _DummyMetric)
        assert len(reg) == 1

    def test_iter(self):
        reg = Registry("test")
        reg.register("a", _DummyMetric)
        reg.register("b", _DummyModel)
        assert set(reg) == {"a", "b"}

    def test_create(self):
        reg = Registry("test")
        reg.register("foo", _DummyMetric)
        instance = reg.create("foo", alpha=2.0)
        assert isinstance(instance, _DummyMetric)
        assert instance.alpha == 2.0

    def test_create_from_config(self):
        reg = Registry("test")
        reg.register("foo", _DummyMetric)
        instance = reg.create_from_config({"name": "foo", "alpha": 3.0})
        assert instance.alpha == 3.0

    def test_create_from_config_type_key(self):
        reg = Registry("test")
        reg.register("foo", _DummyMetric)
        instance = reg.create_from_config({"type": "foo", "alpha": 5.0})
        assert instance.alpha == 5.0

    def test_create_from_config_no_name_raises(self):
        reg = Registry("test")
        with pytest.raises(ValueError, match="name"):
            reg.create_from_config({"alpha": 1.0})

    def test_get_info(self):
        reg = Registry("test")
        reg.register("a", _DummyMetric, category="info", description="Test metric", tags=["tag1"])
        info = reg.get_info("a")
        assert isinstance(info, ComponentInfo)
        assert info.category == "info"
        assert "tag1" in info.tags

    def test_get_metadata(self):
        reg = Registry("test")
        reg.register("a", _DummyMetric, category="info", description="desc")
        meta = reg.get_metadata("a")
        assert meta["category"] == "info"

    def test_search_by_query(self):
        reg = Registry("test")
        reg.register("rayleigh_quotient", _DummyMetric, description="alignment metric")
        reg.register("magnitude", _DummyModel, description="weight pruning")
        results = reg.search(query="alignment")
        assert "rayleigh_quotient" in results
        assert "magnitude" not in results

    def test_search_by_tags(self):
        reg = Registry("test")
        reg.register("a", _DummyMetric, tags=["info", "fast"])
        reg.register("b", _DummyModel, tags=["info", "slow"])
        results = reg.search(tags=["info", "fast"])
        assert "a" in results
        assert "b" not in results

    def test_summary(self):
        reg = Registry("test")
        reg.register("a", _DummyMetric, category="info")
        summary = reg.summary()
        assert "info" in summary
        assert "a" in summary

    def test_overwrite_warns(self):
        reg = Registry("test")
        reg.register("a", _DummyMetric)
        # Registering again should warn, not raise
        reg.register("a", _DummyModel)
        assert reg.get("a") is _DummyModel


# =========================================================================
# Module-level helpers
# =========================================================================


class TestModuleLevelHelpers:
    def test_create_component_unknown_registry(self):
        with pytest.raises(KeyError, match="Unknown registry"):
            create_component("bogus_registry", "foo")

    def test_create_from_config_no_registry(self):
        with pytest.raises(ValueError, match="registry"):
            create_from_config({"name": "foo"})

    def test_list_all_components_returns_dict(self):
        result = list_all_components()
        assert isinstance(result, dict)
        assert "metrics" in result
