# Write Vault Note

You are tasked with creating a high-quality Obsidian notes plan for the vault through an interactive, plan-first process that leverages parallel sub-agents to research and gather information.

## Initial Response

When this command is invoked:

1. **Check if parameters were provided**:
   - If a topic or file path was provided as a parameter, skip the default message
   - Immediately begin the research process

2. **If no parameters provided**, respond with:
```
I'll help you create a vault note plan. Let me start by understanding what you want to document.

Please provide:
- The topic or concept you want documented
- The type of note (concept/technology/example)
- Any specific aspects, relationships, or sources to explore
- Links to related research, documentation, or existing notes

I'll research the topic and work with you to create a comprehensive note outline before writing.

Tip: You can also invoke this command with a topic directly: `/plan_vault_note Context Windows in LLMs`
```

Then wait for the user's input.

## Process Steps

### Step 1: Context Gathering & Initial Research

1. **Read all mentioned files immediately and FULLY**:
   - Any existing vault notes, research documents, or source materials mentioned
   - **IMPORTANT**: Use the Read tool WITHOUT limit/offset parameters to read entire files
   - **CRITICAL**: DO NOT spawn sub-tasks before reading these files yourself in the main context
   - **NEVER** read files partially - if a file is mentioned, read it completely

2. **Spawn initial research tasks to gather context**:
   Before asking the user any questions, use specialized agents to research in parallel:

   - Use the **thoughts-locator** agent to find any existing research on this topic
   - Use the **codebase-pattern-finder** agent to find implementation examples if applicable
   - Check the vault for similar or related notes that already exist

3. **Analyze the topic and identify knowledge gaps**:
   - Determine what information is available vs. what needs to be researched
   - Identify key aspects: definition, features, relationships, examples, use cases
   - Note which existing vault notes should be linked
   - Determine what external information is needed (web search, documentation)

4. **Present informed understanding and focused questions**:
   ```
   Based on my initial research, I understand [topic] is [brief summary].

   I've found:
   - [Existing vault notes that relate to this]
   - [Internal research or implementation examples]
   - [Initial understanding of the concept]

   Questions before I proceed:
   - [Specific aspect that needs clarification]
   - [Scope question - what to include/exclude]
   - [Relationship to other concepts]
   ```

   Only ask questions that you genuinely cannot answer through research.

### Step 2: Deep Research & Discovery

After getting initial clarifications:

1. **Create a research todo list** using TodoWrite to track exploration tasks

2. **Spawn parallel sub-agent tasks for comprehensive research**:
   - Create multiple Task agents to research different aspects concurrently

   Use specialized agents intelligently:
   - **web-search-researcher**: For current information, documentation, tutorials, examples
   - **codebase-locator**: To find related implementation examples in your codebase
   - **codebase-analyzer**: For deep analysis of implementation patterns
   - **thoughts-locator**: To find related research and notes
   - **codebase-pattern-finder**: To discover usage patterns and examples

   Example parallel research tasks:
   - Agent 1: Web search for official documentation and key features
   - Agent 2: Web search for real-world examples and use cases
   - Agent 3: Search vault for related concepts and potential links
   - Agent 4: Search codebase for implementation examples (if applicable)
   - Agent 5: Search thoughts for existing research on the topic

3. **Wait for ALL sub-tasks to complete** before proceeding

4. **Synthesize findings and identify gaps**:
   - Compile all findings from web research, vault, codebase, and thoughts
   - Identify key concepts, features, and relationships
   - Determine appropriate links to other vault notes using [[wikilink]] format
   - Extract concrete examples and use cases
   - Verify all information is accurate and up-to-date
   - Note any remaining open questions or uncertainties

5. **If uncertainties remain**:
   - STOP and ask for clarification
   - Do NOT proceed to outlining with unresolved questions
   - Spawn additional research tasks if needed

### Step 3: Note Outline Development

Once research is complete and all questions are answered:

1. **Determine note type and structure**:
   - Confirm whether this should be a concept, technology, or example note
   - Verify the structure matches the content gathered

2. **Create initial note outline with content summary**:
   ```
   Here's my proposed note structure for [Topic Name]:

   **Type**: [concept/technology/example]
   **File**: `apps/vault/[type]/[filename].md`

   ## Sections to include:

   ### 1. [Section Name]
   - [Key point to cover]
   - [Another key point]
   - Links to: [[related-note-1]], [[related-note-2]]

   ### 2. [Section Name]
   - [Key point to cover]
   - [Another key point]
   - Example: [specific example found in research]

   ### 3. [Section Name]
   - [Key point to cover]
   - Resources: [documentation links found]

   ## Key Relationships:
   - Links to existing notes: [[note1]], [[note2]], [[note3]]
   - Technologies mentioned: [[tech1]], [[tech2]]
   - Concepts referenced: [[concept1]], [[concept2]]

   ## Open Questions:
   - [Any remaining uncertainties if applicable]

   Does this structure capture what you want? Should I adjust any sections or add/remove content areas?
   ```

3. **Get feedback on structure** before writing the note

4. **If user suggests changes**:
   - Update the outline based on feedback
   - If changes require more research, spawn additional research tasks
   - Get approval on the revised outline

### Step 4: Note Writing

After outline approval:

1. **Write the complete note** using the approved structure
2. **Follow the template for the note type**:
   - Include YAML frontmatter with correct type and tags
   - Use [[wikilinks]] for all references to other vault notes
   - Keep language clear, concise, and technical
   - Include concrete examples where applicable
   - Add proper hierarchy with markdown headings
   - Include changelog entry with today's date
   - Save to appropriate directory: `apps/vault/concepts/`, `apps/vault/tech/`, or `apps/vault/examples/`
   - Use kebab-case for filenames (e.g., `prompt-scaffolding.md`, `claude-code.md`)

3. **Verify note completeness**:
   - All sections from the approved outline are included
   - All [[wikilinks]] point to existing notes (or note if they're placeholders)
   - Examples and use cases are concrete and clear
   - Resources and references are accurate

### Step 5: Index Update & Presentation

After writing the note:

1. **Update vault index if needed:**
   - If this is a significant new concept, technology, or example
   - Add entry to `apps/vault/README.md` in appropriate section
   - Maintain existing structure and formatting

2. **Present the completed note**:
   ```
   I've created the vault note at:
   `apps/vault/[type]/[filename].md`

   **Summary:**
   - [Brief description of what was documented]
   - [Key sections included]
   - [Number] relationships established via [[wikilinks]]

   **Key Relationships:**
   - Links to: [[note1]], [[note2]], [[note3]]
   - Referenced by concepts: [list]

   Would you like me to:
   - Expand any sections with more detail?
   - Create related notes for any linked concepts?
   - Add more examples or use cases?
   ```

### Step 6: Iteration & Refinement

1. **Handle follow-up requests**:
   - If user wants to expand sections, spawn research tasks for deeper investigation
   - Update the `updated` field in frontmatter
   - Add new changelog entry
   - Maintain consistency with existing structure

2. **For additional related notes**:
   - Start the process over for each new note
   - Ensure bidirectional links between related notes
   - Maintain consistency in terminology and structure

## Important Guidelines

1. **Be Interactive**:
   - Don't write the note in one shot
   - Get buy-in on the outline first
   - Allow course corrections
   - Work collaboratively

2. **Be Thorough**:
   - Read all context files COMPLETELY before planning
   - Research using parallel sub-tasks for efficiency
   - Verify information from official documentation and primary sources
   - Include specific examples and practical applications
   - Ensure all [[wikilinks]] point to existing notes (or clearly mark as placeholders)

3. **Be Skeptical**:
   - Question vague requirements
   - Verify understanding through research
   - Don't assume - investigate first
   - Identify missing information early

4. **Track Progress**:
   - Use TodoWrite to track research and writing tasks
   - Update todos as you complete research phases
   - Mark tasks complete when done

5. **No Open Questions in Final Note**:
   - If you encounter open questions during research, STOP
   - Research or ask for clarification immediately
   - Do NOT write the note with unresolved questions
   - The note must be complete and accurate before finalizing

## Note Structure Templates

**For Concept Notes** (e.g., prompt-scaffolding, context-engineering):
```markdown
---
title: [Concept Name]
type: concept
tags: [relevant, tags, here]
created: YYYY-MM-DD
updated: YYYY-MM-DD
---

# [Concept Name]

## Definition
[Clear, concise definition]

## Key Techniques/Components
[Main elements of the concept]

## Best Practices
[Guidelines for applying the concept]

## How It Relates
[Links to related concepts using [[wikilinks]]]

## Key Technologies
[Technologies that implement or support this concept]

## Real-World Applications
[Practical use cases and examples]

---

## Changelog
- **YYYY-MM-DD**: [Change description]
```

**For Technology Notes** (e.g., claude-code, langfuse):
```markdown
---
title: [Technology Name]
type: technology
category: [development/orchestration/observability/data/infrastructure]
tags: [relevant, tags, here]
created: YYYY-MM-DD
updated: YYYY-MM-DD
website: [URL]
github: [URL if applicable]
---

# [Technology Name]

## Overview
[Brief description and purpose]

## Key Features
[Main capabilities and features]

## Architecture/How It Works
[Technical details, workflow, or architecture]

## Use Cases in Context Engineering
[How it fits into the context engineering ecosystem]

## Related Technologies
[Links to other tech notes using [[wikilinks]]]

## Resources
[Official docs, tutorials, examples]

---

## Changelog
- **YYYY-MM-DD**: [Change description]
```

**For Example Notes** (e.g., agentic-rag-workflow):
```markdown
---
title: [Example Name]
type: example
tags: [relevant, tags, here]
created: YYYY-MM-DD
updated: YYYY-MM-DD
---

# [Example Name]

## Overview
[What this example demonstrates]

## Architecture
[Diagram or description of components]

## Key Components
[Main pieces with links to related notes]

## Implementation Steps/Workflow
[How it works step by step]

## Context Engineering Strategies
[Which strategies from [[context-engineering]] are used]

## Benefits
[Advantages of this approach]

## Related Concepts
[Links to relevant concept and tech notes]

---

## Changelog
- **YYYY-MM-DD**: [Change description]
```

## Template Guidelines

- **File reading**: Always read mentioned files FULLY (no limit/offset) before spawning sub-tasks
- **Critical ordering**: Follow the numbered steps exactly
  - ALWAYS read mentioned files first before spawning sub-tasks (Step 1)
  - ALWAYS wait for all sub-agents to complete before synthesizing (Step 2)
  - ALWAYS create outline and get approval before writing (Step 3 before Step 4)
  - NEVER write the note with incomplete research or unapproved outline
- **Link validation**: Only create [[wikilinks]] to notes that exist or will be created
- **Frontmatter consistency**:
  - Always include frontmatter at the beginning
  - Use consistent field names across note types
  - Tags should be array format: [tag1, tag2, tag3]
  - Type must be one of: concept, technology, example, index
- **Vault structure preservation**:
  - concepts/ for conceptual frameworks and methodologies
  - tech/ for specific technologies and platforms
  - examples/ for implementation examples and patterns
  - Maintain existing categorization patterns
- Tags should be lowercase and relevant to the content
- Dates in YYYY-MM-DD format
- Use proper markdown hierarchy (##, ###, etc.)
- Keep definitions clear and accessible
