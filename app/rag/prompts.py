"""Versioned prompt templates: configs/prompts/<name>_v<version>.yaml (never edit old versions)."""

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel


class PromptTemplate(BaseModel):
    name: str
    version: int
    system: str
    user: str
    refusal_text: str | None = None

    def render_system(self, **values: str) -> str:
        return self.system.format(refusal_text=self.refusal_text or "", **values)

    def render_user(self, **values: str) -> str:
        return self.user.format(**values)


@lru_cache
def load_prompt(prompts_dir: Path, name: str, version: int) -> PromptTemplate:
    path = prompts_dir / f"{name}_v{version}.yaml"
    with path.open(encoding="utf-8") as f:
        template = PromptTemplate.model_validate(yaml.safe_load(f))
    if template.name != name or template.version != version:
        raise ValueError(f"{path} declares {template.name} v{template.version}")
    return template
