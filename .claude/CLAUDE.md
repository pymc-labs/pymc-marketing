# PyMC-Marketing — Claude Code Instructions

## Environment

- Activate conda env `pymc-marketing-dev` before running any commands.
- Run `pre-commit` every time you create or modify a file.
- Temporary or test scripts that are not meant to be committed should be created in the `sandbox/` folder.

## Skills (shared with Cursor)

Domain knowledge and workflow skills live in `.cursor/skills/`. Read the relevant
skill file **before** starting work in that domain.

| Skill | When to read | Entry file |
|---|---|---|
| **mmm-modeling** | Building, fitting, diagnosing, or optimizing Media Mix Models with PyMC-Marketing | `.cursor/skills/mmm-modeling/SKILL.md` (references in `references/` subdirectory) |
| **code-best-practice** | Writing or reviewing PyMC-Marketing code (style, typing, docstrings, PyTensor, testing) | `.cursor/skills/code-best-practice/SKILL.md` |
| **research** | Structuring a research task before implementation | `.cursor/skills/research/SKILL.md` |
| **make-plan** | Creating an implementation plan from research | `.cursor/skills/make-plan/SKILL.md` |
| **implement** | Implementing code from a plan | `.cursor/skills/implement/SKILL.md` |
| **commit** | Creating git commits for session changes | `.cursor/skills/commit/SKILL.md` |

### How to use skills

1. When a task matches a skill's trigger (the "When to read" column), read the
   entry file in full.
2. The entry file may reference additional files (e.g., `references/model_fit.md`).
   Read those as needed for the specific sub-topic.
3. Follow the patterns and constraints described in the skill.
