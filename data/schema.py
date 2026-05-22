from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.schema import CreateColumn, MetaData


@dataclass(frozen=True)
class MissingColumn:
    table: str
    column: str
    nullable: bool
    can_auto_add: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table,
            "column": self.column,
            "nullable": self.nullable,
            "can_auto_add": self.can_auto_add,
        }


def ensure_schema(engine: Engine, metadata: MetaData) -> dict[str, Any]:
    """Create tables and add safe nullable columns for old SQLite databases."""
    metadata.create_all(engine)
    report = schema_report(engine, metadata)
    auto_add = [
        MissingColumn(**item)
        for item in report["missing_columns"]
        if item["can_auto_add"]
    ]
    if auto_add:
        with engine.begin() as connection:
            for missing in auto_add:
                table = metadata.tables[missing.table]
                column = table.columns[missing.column]
                ddl = CreateColumn(column).compile(dialect=engine.dialect)
                connection.execute(text(f'ALTER TABLE "{missing.table}" ADD COLUMN {ddl}'))
        report = schema_report(engine, metadata)
    return report


def schema_report(engine: Engine, metadata: MetaData) -> dict[str, Any]:
    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    expected_tables = [table.name for table in metadata.sorted_tables]
    missing_tables = [name for name in expected_tables if name not in existing_tables]
    missing_columns: list[dict[str, Any]] = []
    table_reports: list[dict[str, Any]] = []

    for table in metadata.sorted_tables:
        if table.name not in existing_tables:
            table_reports.append(
                {
                    "table": table.name,
                    "status": "missing",
                    "missing_columns": [column.name for column in table.columns],
                }
            )
            continue
        existing_columns = {column["name"] for column in inspector.get_columns(table.name)}
        table_missing: list[str] = []
        for column in table.columns:
            if column.name in existing_columns:
                continue
            can_auto_add = _can_auto_add_column(column)
            missing = MissingColumn(
                table=table.name,
                column=column.name,
                nullable=bool(column.nullable),
                can_auto_add=can_auto_add,
            )
            missing_columns.append(missing.to_dict())
            table_missing.append(column.name)
        table_reports.append(
            {
                "table": table.name,
                "status": "ok" if not table_missing else "missing_columns",
                "missing_columns": table_missing,
            }
        )

    unmanaged = [item for item in missing_columns if not item["can_auto_add"]]
    return {
        "ok": not missing_tables and not missing_columns,
        "missing_tables": missing_tables,
        "missing_columns": missing_columns,
        "unmanaged_missing_columns": unmanaged,
        "tables": table_reports,
    }


def _can_auto_add_column(column: Any) -> bool:
    if column.primary_key:
        return False
    return bool(column.nullable or column.server_default is not None)
