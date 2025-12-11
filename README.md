# Tool Selection Pipeline

This repo contains a retrieval-first tool selection system with optional LLM rerank and planning. The flow you’ll see in the UI:

1) Landing page (enter your query, pick backend, toggle streaming)
<p align="center">
  <img src="docs/landing.png" alt="Landing page" width="75%">
</p>

2) Send a query (e.g., `Book a flight from LA to NYC on June 15th`) — first response shows the top candidate tools
<p align="center">
  <img src="docs/candidates.png" alt="Candidate tools" width="75%">
</p>

3) Then view the generated plan (steps) for the chosen tool(s)
<p align="center">
  <img src="docs/steps.png" alt="Plan steps" width="75%">
</p>

You may read the detailed documentation:

- [Setup](docs/setup.md) — how to install and run locally (Poetry) or via Docker.
- [Workflow](docs/workflow.md) — pipeline flow diagram and component one-liners.
- [Modules](docs/modules.md) — what each module does and why it exists.
- [Design](docs/design.md) — design rationale and improvements over full-context prompting.
