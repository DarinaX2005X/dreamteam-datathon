"""Generate deterministic JSON Schema, OpenAPI and TypeScript from Pydantic.

Run: python -m navidu.contracts.generate. This small JSON Schema renderer supports
only the schema constructs we emit; unsupported constructs fail generation.
"""

import inspect
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, TypeAdapter

from . import api_v1, domain, errors, pipeline, rules, ui_api

ROOT = Path(__file__).resolve().parents[3] / "contracts"
MODULES = [domain, errors, api_v1, pipeline, rules, ui_api]


def models() -> dict[str, type[BaseModel]]:
    return {
        name: value
        for module in MODULES
        for name, value in vars(module).items()
        if inspect.isclass(value)
        and issubclass(value, BaseModel)
        and value.__module__ == module.__name__
    }


def typescript(schema: dict[str, Any]) -> str:
    if "$ref" in schema:
        return schema["$ref"].split("/")[-1]
    if "const" in schema:
        return json.dumps(schema["const"], ensure_ascii=False)
    if "enum" in schema:
        return " | ".join(json.dumps(value, ensure_ascii=False) for value in schema["enum"])
    for key in ["anyOf", "oneOf"]:
        if key in schema:
            return " | ".join(typescript(value) for value in schema[key])
    if "allOf" in schema:
        return " & ".join(typescript(value) for value in schema["allOf"])
    kind = schema.get("type")
    if kind == "array":
        if "prefixItems" in schema:
            return "[" + ", ".join(typescript(value) for value in schema["prefixItems"]) + "]"
        return f"Array<{typescript(schema.get('items', {}))}>"
    if kind == "object":
        required = set(schema.get("required", []))
        fields = [
            f"  {json.dumps(key)}{'' if key in required else '?'}: {typescript(value)};"
            for key, value in schema.get("properties", {}).items()
        ]
        additional = schema.get("additionalProperties")
        if additional is True:
            fields.append("  [key: string]: unknown;")
        elif isinstance(additional, dict):
            fields.append(f"  [key: string]: {typescript(additional)};")
        return "{\n" + "\n".join(fields) + "\n}"
    if kind in ["number", "integer"]:
        return "number"
    if kind in ["string", "boolean", "null"]:
        return kind
    if not schema or not kind:
        return "unknown"
    raise ValueError(f"Unsupported JSON Schema: {schema}")


def generate() -> dict[str, Any]:
    schemas: dict[str, Any] = {}
    individual = {}
    for name, cls in sorted(models().items()):
        schema = cls.model_json_schema(ref_template="#/components/schemas/{model}")
        schemas.update(schema.pop("$defs", {}))
        schemas[name] = schema
        individual[name] = cls.model_json_schema()
    for name, value in {
        "Role": domain.Role,
        "ZoneKind": domain.ZoneKind,
        "ObservationKind": domain.ObservationKind,
        "Verdict": domain.Verdict,
        "DecisionValue": domain.DecisionValue,
        "CheckStatus": ui_api.CheckStatus,
        "PipelinePhase": pipeline.PipelinePhase,
        "JobResponse": api_v1.JobResponse,
    }.items():
        schema = TypeAdapter(value).json_schema(ref_template="#/components/schemas/{model}")
        schemas.update(schema.pop("$defs", {}))
        schemas[name] = schema
        individual[name] = TypeAdapter(value).json_schema()

    def ref(name):
        return {"$ref": f"#/components/schemas/{name}"}

    def operation(response, request=None, upload=False, jury=False, status="200"):
        result: dict[str, Any] = {"responses": {
            status: {"description": "Success", "content": {"application/json": {
                "schema": ref(response)
            }}},
            "default": {"description": "Error", "content": {"application/json": {
                "schema": ref("ErrorResponse")
            }}},
        }}
        if request:
            result["requestBody"] = {"required": True, "content": {
                "application/json": {"schema": ref(request)}
            }}
        if upload:
            result["requestBody"] = {"required": True, "content": {"multipart/form-data": {
                "schema": {"type": "object", "required": ["file"], "properties": {
                    "file": {"type": "string", "format": "binary"}
                }}
            }}}
        if jury:
            result["security"] = [{"ApiKey": []}]
        return result

    paths = {
        "/api/v1/analyze": {"post": operation("JobProcessing", upload=True, jury=True)},
        "/api/v1/jobs/{job_id}": {"get": operation("JobResponse", jury=True)},
        "/api/ui/checks": {"post": operation("CheckSummaryResponse", upload=True, status="201")},
        "/api/ui/checks/{id}": {"get": operation("CheckDetailResponse")},
        "/api/ui/checks/{id}/findings/{finding_id}/decision": {
            "put": operation("DecisionResponse", "DecisionRequest")
        },
        "/api/ui/checks/{id}/reevaluate": {
            "post": operation("ReevaluateResponse", "ReevaluateRequest")
        },
        "/api/ui/checks/{id}/retry": {"post": operation("CheckSummaryResponse")},
        "/api/ui/checks/{id}/cancel": {"post": operation("CheckSummaryResponse")},
        "/api/ui/rules": {"get": operation("RulesResponse")},
        "/api/ui/example": {"get": operation("ExampleResponse")},
        "/api/ui/health": {"get": operation("HealthResponse")},
    }
    for path, mime in [
        ("/api/ui/checks/{id}/video", "video/mp4"),
        ("/api/ui/checks/{id}/frames/{frame_id}.jpg", "image/jpeg"),
    ]:
        paths[path] = {"get": {"responses": {"200": {"description": "Local media", "content": {
            mime: {"schema": {"type": "string", "format": "binary"}}
        }}}}}
    import re
    for path, methods in paths.items():
        for operation_value in methods.values():
            names = re.findall(r"\{([^}]+)\}", path)
            if names:
                operation_value["parameters"] = [
                    {"name": name, "in": "path", "required": True, "schema": {"type": "string"}}
                    for name in names
                ]
    paths["/api/ui/checks/{id}/frames/{frame_id}.jpg"]["get"]["parameters"].extend([
        {"name": "overlay", "in": "query", "schema": {"type": "integer", "enum": [0, 1], "default": 1}},
        {"name": "revision", "in": "query", "schema": {"type": "integer", "minimum": 1},
         "description": "Finding-set revision; defaults to the current persisted revision."},
    ])
    document = {
        "openapi": "3.1.0",
        "info": {"title": "Lado local video analysis", "version": "0.1.0"},
        "paths": paths,
        "components": {"schemas": schemas, "securitySchemes": {
            "ApiKey": {"type": "apiKey", "in": "header", "name": "X-API-Key"}
        }},
    }
    (ROOT / "schemas").mkdir(parents=True, exist_ok=True)
    (ROOT / "ts").mkdir(parents=True, exist_ok=True)
    for name, value in sorted(individual.items()):
        (ROOT / "schemas" / f"{name}.json").write_text(
            json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    (ROOT / "openapi.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / "ts" / "types.gen.ts").write_text(
        "// Generated by python -m navidu.contracts.generate. Do not edit.\n\n" + "\n\n".join(
            f"export type {name} = {typescript(value)};" for name, value in sorted(schemas.items())
        ) + "\n", encoding="utf-8"
    )
    return document


if __name__ == "__main__":
    generate()
    print(f"Generated contracts in {ROOT}")
