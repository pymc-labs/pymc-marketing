---
name: make-plan
description: Create a plan based on document research through an interactive, iterative process.
disable-model-invocation: true
---

# Create a plan based on document research
You are tasked with creating detailed implementation plans through an interactive, iterative process. You should be skeptical, thorough, and work collaboratively with the user to produce high-quality technical specifications.

Reading the discoveries, during the research, make a plan in `.cursor/plans/{plan_name_folder}` using a `.md` file style.

You must:
- Define a to-do list.
- If any information is missing, ask the user.
- DO NOT just accept the correction or suggestion from user.
- Spawn new research tasks to verify the correct information (research/command).
- Read the specific files/directories they mentioned.
- Only proceed once you've verified the facts yourself.
