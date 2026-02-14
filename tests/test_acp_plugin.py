"""Tests for the ACP plugin."""

from __future__ import annotations

import click

from pykoclaw_acp import AcpPlugin


def test_acp_plugin_implements_protocol() -> None:
    from pykoclaw.plugins import PykoClawPlugin

    plugin = AcpPlugin()
    assert isinstance(plugin, PykoClawPlugin)


def test_register_commands_adds_acp_command() -> None:
    plugin = AcpPlugin()
    group = click.Group()

    plugin.register_commands(group)

    assert "acp" in group.commands
    assert isinstance(group.commands["acp"], click.Command)


def test_acp_plugin_default_methods() -> None:
    import sqlite3

    plugin = AcpPlugin()

    db = sqlite3.connect(":memory:")
    assert plugin.get_mcp_servers(db, "test") == {}
    assert plugin.get_db_migrations() == []
    assert plugin.get_config_class() is None
